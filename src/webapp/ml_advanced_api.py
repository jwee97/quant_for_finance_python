"""Release 8 routes: the optional advanced modules that work on saved experiments and the registry's records (spec v3, P3-01, P3-02, P3-10, P3-11, P3-12).

These are mixed into ``MlApi``. They are off by default in the only sense that matters for a service: nothing here is scheduled and nothing runs unless a request names it, and none of the advanced
modules is imported until one does. Every route delegates to a service function in ``src.ml`` (the same one the command line and programmatic callers use), refuses options it does not take, names the
person behind every decision, and returns ``null`` plus a reason for evidence it does not have. The numerical modules (Bayesian averaging, robust allocation, horizon combination, transfer, discovery,
off-policy evaluation, tax lots) take arrays the caller supplies and have no route; ``GET /api/ml/advanced`` says so.
"""

from __future__ import annotations

from dataclasses import asdict

from ..assets import common
from ..ml import advanced as adv
from ..ml import diagnostics as dx
from .api import ApiError
from .payload import clean

ROLES = ("champion", "challenger")


def _h():
    """The helpers of ``ml_api`` (imported when a request arrives, because ``ml_api`` imports this module)."""
    from . import ml_api

    return ml_api


def _wrap(kind: str, out: dict, notes=()) -> dict:
    """The response envelope. ``as_of`` is a provenance field of the envelope, so a report's own ``as_of`` moves there instead of clashing with it."""
    out = dict(out)
    as_of = out.pop("as_of", None)
    return clean(common.envelope(kind, out, {"registry": "platform"}, [], as_of, list(notes)))


def _actor(body: dict, required: bool = True) -> str:
    actor = str(body.get("actor") or "").strip()
    if required and not actor:
        raise ApiError("actor is required: a decision names the person who made it")
    return actor[:80] or "api"


def _text(body: dict, key: str, limit: int = 500, required: bool = True) -> str:
    value = str(body.get(key) or "").strip()
    if required and not value:
        raise ApiError(f"{key} is required")
    return value[:limit]


class AdvancedRoutes:
    def _stored_advanced(self, kind: str, exp_id: str, record_kind: str) -> dict:
        """The latest record of a kind, as stored: reading it computes and records nothing."""
        rec = dx.latest(self.registry, exp_id, record_kind)
        out = ({"eligibility": "not_computed", "reason": "these diagnostics have not been computed for this experiment", "experiment_id": exp_id} if rec is None
               else {**rec["content"], "record_seq": rec["seq"], "computed_at": rec["created"], "stored": True})
        return _wrap(kind, out)

    def advanced_get_routes(self) -> dict:
        return {"/api/ml/advanced": self.advanced_catalog, "/api/ml/drift": self.drift, "/api/ml/retraining": self.retraining, "/api/ml/champion": self.champion,
                "/api/ml/uncertainty-monitor": self.uncertainty_monitor, "/api/ml/reports": self.reports, "/api/ml/report": self.report}

    def advanced_post_routes(self) -> dict:
        return {"/api/ml/retraining/action": self.retraining_action, "/api/ml/champion/register": self.champion_register, "/api/ml/champion/commit": self.champion_commit,
                "/api/ml/champion/look": self.champion_look, "/api/ml/report/build": self.report_build}

    # ------------------------------------------------------------------------------------------------------------ the catalogue
    def advanced_catalog(self, q: dict) -> dict:
        """``GET /api/ml/advanced``: the twelve optional advanced modules, each off by default, with the way it is reached (API, command line or library), what it needs and whether it can run here, beside the
        latest recorded verification of its requirement."""
        h = _h()
        h._only(q, set(), "/api/ml/advanced")
        reg = self.registry if (self.registry_root / "registry.sqlite").exists() else None
        return clean(common.envelope("ml_advanced", adv.catalog(reg), {"registry": "platform"}, [], None,
                                     ["Nothing here runs unless a request names it. 'Available' means the code runs in this environment, not that the technique helps."]))

    # ------------------------------------------------------------------------------------------------------------ P3-01 drift
    def drift(self, q: dict) -> dict:
        """``GET /api/ml/drift?id=<experiment>[&as_of=<date>][&reference_days=504][&recent_days=63][&psi_watch=0.1][&psi_alert=0.25][&correction=bh][&min_coverage=0.8][&stored=1]``: the input drift of the
        experiment's features against the fixed window that ends at its model's fit cutoff (PSI, a dependence-aware KS test adjusted across features, sample counts and coverage) and the drift of its settled
        rank IC, with the link to the previous look. An alert is a proposal to look, not an action. Each request is recorded; ``stored=1`` reads the latest record and computes nothing."""
        h = _h()
        h._only(q, {"id", "as_of", "reference_days", "recent_days", "psi_watch", "psi_alert", "correction", "min_coverage", "stored"}, "/api/ml/drift")
        with h._user_errors():
            exp_id = h._id(q)
            self.registry.get_experiment(exp_id)
            if str(q.get("stored") or "") in ("1", "true"):
                return self._stored_advanced("ml_drift", exp_id, "drift_report")
            from ..ml.drift import DriftSpec

            spec = DriftSpec(reference_days=h._int(q, "reference_days", 504, 20, 5000), recent_days=h._int(q, "recent_days", 63, 5, 1000), psi_watch=h._float(q, "psi_watch", 0.10, 0.0, 5.0),
                             psi_alert=h._float(q, "psi_alert", 0.25, 0.0, 5.0), correction=str(q.get("correction") or "bh"), min_coverage=h._float(q, "min_coverage", 0.8, 0.0, 1.0))
            as_of = h._date("as_of", q["as_of"]) if q.get("as_of") else None
            with self._heavy:
                out = adv.run_drift(self.registry, exp_id, spec=spec, as_of=as_of, actor="api")
        return _wrap("ml_drift", out)

    # ------------------------------------------------------------------------------------------------------------ P3-02 retraining
    def retraining(self, q: dict) -> dict:
        """``GET /api/ml/retraining?id=<experiment>``: the retraining history of an experiment (every trigger with the reason it was or was not a proposal, approvals, rejections, failures, suspension,
        recovery, rollback) and the state it leaves: the open proposal, consecutive failures and whether it is suspended."""
        h = _h()
        h._only(q, {"id"}, "/api/ml/retraining")
        from ..ml import retrain as rt

        with h._user_errors():
            exp_id = h._id(q)
            self.registry.get_experiment(exp_id)
            events = rt.events(self.registry, exp_id)
        data = {"subject": exp_id, "state": rt.status(self.registry, exp_id), "events": events[-60:], "events_total": len(events), "triggers": list(rt.TRIGGERS), "default_policy": asdict(rt.RetrainPolicy(name="default")),
                "note": "A trigger is a proposal. Nothing is trained until a person approves it (or a policy allows it, from code), a retraining cannot read a protected segment, and a challenger is never promoted by it."}
        return clean(common.envelope("ml_retraining", data, {"registry": "platform"}, [], None, []))

    def retraining_action(self, body: dict) -> dict:
        """``POST /api/ml/retraining/action {"id", "action": "trigger"|"approve"|"reject"|"recover"|"rollback", ...}``. ``trigger`` takes ``kind`` (drift_alert and performance_alert need an alert in the latest
        drift report; schedule and manual are accepted as declared) and an optional ``policy``; ``approve`` and ``reject`` take ``proposal``, ``actor`` and ``note``; ``recover`` lifts a suspension (``actor``,
        ``note``); ``rollback`` takes ``from_version``, ``to_version``, ``actor`` and ``note`` and goes through governance. Starting a retraining is not a route: it needs a trainer and runs from code."""
        h = _h()
        h._only(body, {"id", "action", "kind", "policy", "proposal", "actor", "note", "from_version", "to_version"}, "/api/ml/retraining/action")
        from ..ml import retrain as rt

        action = str(body.get("action") or "")
        if action not in ("trigger", "approve", "reject", "recover", "rollback"):
            raise ApiError("action is one of trigger, approve, reject, recover, rollback")
        with h._user_errors():
            exp_id = h._id(body)
            self.registry.get_experiment(exp_id)
            try:
                if action == "trigger":
                    policy = adv.retrain_policy(body.get("policy"))
                    with self._heavy:
                        out = adv.propose_retraining(self.registry, exp_id, kind=str(body.get("kind") or "manual"), policy=policy, actor=_actor(body, required=False))
                elif action in ("approve", "reject"):
                    proposal = h._num("proposal", body.get("proposal"), 1, 10**9, True)
                    out = (rt.approve if action == "approve" else rt.reject)(self.registry, exp_id, proposal, actor=_actor(body), note=_text(body, "note"))
                elif action == "recover":
                    out = rt.recover(self.registry, exp_id, actor=_actor(body), note=_text(body, "note"))
                else:
                    out = rt.rollback(self.registry, exp_id, _text(body, "from_version", 100), _text(body, "to_version", 100), reason=_text(body, "note"), actor=_actor(body))
            except rt.RetrainError as exc:
                raise ApiError(str(exc)) from None
        return clean(common.envelope("ml_retraining_action", {"subject": exp_id, "action": action, **out, "state": rt.status(self.registry, exp_id)}, {"registry": "platform"}, [], None, []))

    # ------------------------------------------------------------------------------------------------------------ P3-10 champion and challenger
    def champion(self, q: dict) -> dict:
        """``GET /api/ml/champion?id=<pair>``: a registered champion/challenger pair: what was frozen (strategy versions, universe, horizon, the schedule of looks and its alpha spending), the settled pairs
        so far, the looks taken with their verdicts, and the next look's requirement."""
        h = _h()
        h._only(q, {"id"}, "/api/ml/champion")
        from ..ml import champion as ch

        with h._user_errors():
            try:
                out = ch.summary(self.registry, h._id(q))
            except ch.ChampionError as exc:
                raise ApiError(str(exc)) from None
        return clean(common.envelope("ml_champion", out, {"registry": "platform"}, [], None, ["A look is a statement about the evidence at that look. Nothing here changes a stage."]))

    def champion_register(self, body: dict) -> dict:
        """``POST /api/ml/champion/register {"champion", "challenger", "universe_version", "horizon", "looks": [..], "alpha"?, "rule"?, "assumptions"?, "data_cutoff"?, "actor"}``: freeze a pair before the
        observations it will be judged on. The pair id is a hash of everything declared; declaring the same pair again returns it unchanged."""
        h = _h()
        h._only(body, {"champion", "challenger", "universe_version", "horizon", "looks", "alpha", "rule", "assumptions", "data_cutoff", "actor"}, "/api/ml/champion/register")
        from ..ml import champion as ch

        looks = body.get("looks")
        if not isinstance(looks, list) or not 1 <= len(looks) <= 12 or any(not isinstance(n, int) or isinstance(n, bool) for n in looks):
            raise ApiError("looks is a list of 1 to 12 whole numbers of settled decisions, strictly increasing, the first at least 10")
        assumptions = body.get("assumptions")
        if assumptions is not None and (not isinstance(assumptions, dict) or len(str(assumptions)) > 5000):
            raise ApiError("assumptions is an object of at most 5000 characters")
        with h._user_errors():
            try:
                out = ch.register(self.registry, champion=h._id(body, "champion"), challenger=h._id(body, "challenger"), universe_version=_text(body, "universe_version", 100), horizon=h._num("horizon", body.get("horizon"), 1, 252, True),
                                  looks=looks, alpha=h._float(body, "alpha", 0.05, 0.001, 0.4999), rule=str(body.get("rule") or "obrien_fleming"), assumptions=assumptions,
                                  data_cutoff=h._date("data_cutoff", body["data_cutoff"]) if body.get("data_cutoff") else None, actor=_actor(body))
            except ch.ChampionError as exc:
                raise ApiError(str(exc)) from None
        return clean(common.envelope("ml_champion_registered", out, {"registry": "platform"}, [], None, []))

    def champion_commit(self, body: dict) -> dict:
        """``POST /api/ml/champion/commit {"pair", "decision_ts", "forecasts": {"champion": {asset: value}, "challenger": {..}}, "data_cutoff", "fit_cutoffs": {..}, "model_created": {..}, "weights"?, "label_start"?,
        "actor"}``: commit both forecasts for one decision time, together, under one data cutoff. A decision before the registration is refused: prospective evidence is never backfilled. Outcomes are settled
        later through the prospective ledger."""
        h = _h()
        h._only(body, {"pair", "decision_ts", "forecasts", "data_cutoff", "fit_cutoffs", "model_created", "weights", "label_start", "actor"}, "/api/ml/champion/commit")
        from ..ml import champion as ch

        def per_role(key: str, kind: str):
            raw = body.get(key)
            if not isinstance(raw, dict) or set(raw) != set(ROLES):
                raise ApiError(f"{key} has exactly 'champion' and 'challenger'")
            for role in ROLES:
                v = raw[role]
                if kind == "map":
                    if not isinstance(v, dict) or not 1 <= len(v) <= 500 or any(not isinstance(k, str) or isinstance(x, bool) or not isinstance(x, (int, float)) or x != x or abs(x) == float("inf") for k, x in v.items()):
                        raise ApiError(f"{key}.{role} is an object of 1 to 500 assets with finite numbers")
                elif not isinstance(v, str) or not v:
                    raise ApiError(f"{key}.{role} is a date or timestamp")
            return raw

        forecasts = per_role("forecasts", "map")
        fit_cutoffs, model_created = per_role("fit_cutoffs", "date"), per_role("model_created", "date")
        weights = per_role("weights", "map") if body.get("weights") is not None else None
        with h._user_errors():
            try:
                out = ch.commit_pair(self.registry, h._id(body, "pair"), decision_ts=_text(body, "decision_ts", 40), forecasts=forecasts, data_cutoff=_text(body, "data_cutoff", 40), fit_cutoffs=fit_cutoffs,
                                     model_created=model_created, weights=weights, label_start=_text(body, "label_start", 40, required=False) or None, actor=_actor(body))
            except ch.ChampionError as exc:
                raise ApiError(str(exc)) from None
        return clean(common.envelope("ml_champion_committed", out, {"registry": "platform"}, [], None, ["Both forecasts are immutable once committed. Outcomes are settled separately."]))

    def champion_look(self, body: dict) -> dict:
        """``POST /api/ml/champion/look {"pair", "actor"}``: take the next predeclared look, if enough decision pairs have settled. The look is tested at its own level of the alpha-spending schedule, recorded,
        and spent."""
        h = _h()
        h._only(body, {"pair", "actor"}, "/api/ml/champion/look")
        from ..ml import champion as ch

        with h._user_errors():
            try:
                out = ch.look(self.registry, h._id(body, "pair"), actor=_actor(body))
            except ch.ChampionError as exc:
                raise ApiError(str(exc)) from None
        return clean(common.envelope("ml_champion_look", out, {"registry": "platform"}, [], None, ["A look is a statement about the evidence at that look. It promotes nothing."]))

    # ------------------------------------------------------------------------------------------------------------ P3-11 uncertainty monitor
    def uncertainty_monitor(self, q: dict) -> dict:
        """``GET /api/ml/uncertainty-monitor?id=<experiment>[&as_of=<date>][&reference_end=<date>][&stored=1]``: calibration of the stated interval, drift of the point forecasts, drift of the stated
        uncertainty and ensemble disagreement, each on its own. A missing estimate is unavailable or not applicable with its reason and never zero uncertainty. Reads the validation segment only. Each request
        is recorded; ``stored=1`` reads the latest record."""
        h = _h()
        h._only(q, {"id", "as_of", "reference_end", "stored"}, "/api/ml/uncertainty-monitor")
        with h._user_errors():
            exp_id = h._id(q)
            self.registry.get_experiment(exp_id)
            if str(q.get("stored") or "") in ("1", "true"):
                return self._stored_advanced("ml_uncertainty_monitor", exp_id, "uncertainty_monitor")
            with self._heavy:
                out = adv.run_monitor(self.registry, exp_id, as_of=h._date("as_of", q["as_of"]) if q.get("as_of") else None,
                                      reference_end=h._date("reference_end", q["reference_end"]) if q.get("reference_end") else None, actor="api")
        return _wrap("ml_uncertainty_monitor", out)

    # ------------------------------------------------------------------------------------------------------------ P3-12 reproducible reports
    def reports(self, q: dict) -> dict:
        """``GET /api/ml/reports?id=<experiment>``: the reports built for an experiment, newest first, each with its hash. A report is never rewritten; a newer one is a new report."""
        h = _h()
        h._only(q, {"id"}, "/api/ml/reports")
        from ..ml import repro_report as rr

        with h._user_errors():
            exp_id = h._id(q)
            self.registry.get_experiment(exp_id)
            rows = rr.list_reports(self.registry, exp_id)
        return clean(common.envelope("ml_reports", {"experiment_id": exp_id, "reports": rows}, {"registry": "platform"}, [], None, []))

    def report(self, q: dict) -> dict:
        """``GET /api/ml/report?report=<id>[&format=markdown][&verify=1]``: a stored report exactly as it was built (nothing is recomputed), as JSON or Markdown. ``verify=1`` adds whether re-rendering the
        stored snapshot reproduces its hash, and which of its inputs (artifact files, diagnostics, configuration, policy) have changed in the registry since."""
        h = _h()
        h._only(q, {"report", "format", "verify"}, "/api/ml/report")
        from ..ml import repro_report as rr

        fmt = str(q.get("format") or "json")
        if fmt not in ("json", "markdown"):
            raise ApiError("format is json or markdown")
        with h._user_errors():
            rid = h._id(q, "report")
            try:
                rec = rr.get(self.registry, rid)
                data = {"report_id": rid, "report_hash": rec["report_hash"], "experiment_id": rec["experiment_id"], "version": rec["version"], "inputs": rec["inputs"]}
                data.update({"markdown": rr.to_markdown(rec["content"])} if fmt == "markdown" else {"content": rec["content"]})
                if str(q.get("verify") or "") in ("1", "true"):
                    data["reproduction"] = rr.reproduce(self.registry, rid)
                    data["inputs_now"] = rr.verify_inputs(self.registry, rid)
            except rr.ReportError as exc:
                raise ApiError(str(exc)) from None
        return clean(common.envelope("ml_report", data, {"registry": "platform"}, [], None, ["A stored report does not change when models or data are updated later."]))

    def report_build(self, body: dict) -> dict:
        """``POST /api/ml/report/build {"id", "actor"?}``: snapshot an experiment's artifact hashes, recorded diagnostics, gates and policy versions, render a report from that snapshot alone and store it
        (append-only). Building again from unchanged inputs returns the same report."""
        h = _h()
        h._only(body, {"id", "actor"}, "/api/ml/report/build")
        from ..ml import repro_report as rr

        with h._user_errors():
            exp_id = h._id(body)
            self.registry.get_experiment(exp_id)
            try:
                with self._heavy:
                    out = rr.build(self.registry, exp_id, actor=_actor(body, required=False) or "api")
            except rr.ReportError as exc:
                raise ApiError(str(exc)) from None
        return clean(common.envelope("ml_report_built", out, {"registry": "platform"}, [], None, []))

