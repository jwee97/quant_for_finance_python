"""Routes of the constrained model selection (spec v3, P1-15 and P2-17), mixed into ``MlApi``.

``POST /api/ml/select`` chooses among saved experiments on their **validation rows only**: eligibility gates, a ranking by the validation net Sharpe (or the policy's primary metric) computed on the pooled
chronological validation returns, a Pareto analysis, and the simplest candidate that is not meaningfully worse than the highest-ranked one. The result is stored (append-only) with the policy hash and can be
read back with ``GET /api/ml/selection``. ``GET /api/ml/selection-policy`` is the documented set of defaults the forms start from. Nothing here promotes a model or reads a protected segment.
"""

from __future__ import annotations

from dataclasses import asdict

from ..assets import common
from ..ml import diagnostics as dx
from ..ml import model_selection as ms
from ..ml import simplicity as sm
from .api import ApiError
from .payload import clean

POLICY_TEXT = {
    "primary_metric": "The metric the candidates are ranked by, computed on the pooled chronological validation returns. The default is the validation net Sharpe ratio.",
    "secondary": "Criteria that break ties and enter the Pareto comparison: drawdown, turnover, stability across validation sub-periods and complexity.",
    "min_validation_days": "Data: the fewest pooled validation days a candidate may have. Fewer is 'insufficient', not 'ineligible'.",
    "min_coverage": "Data: the share of validation days on which at least two assets were scored.", "min_assets": "Data: the fewest assets scored.",
    "max_drawdown": "Risk: the largest peak-to-trough fall of the validation equity, as a fraction.", "max_ann_vol": "Risk: annualised volatility of the validation net returns (off by default).",
    "max_gross_exposure": "Exposure: the largest sum of absolute weights on any validation day.", "max_abs_net_exposure": "Exposure: the largest absolute sum of weights on any validation day.",
    "max_ann_turnover": "Liquidity: annualised one-way turnover (off by default).",
    "min_capacity_share": "Liquidity: the share of the book one session's participation limit lets through at capacity_capital. Capacity is measured (realistic execution with volume) or unavailable; it is never assumed.",
    "capacity_capital": "Liquidity: the capital level at which min_capacity_share is read.", "min_effective_obs": "Statistical: n(1 - r)/(1 + r) for the lag-1 autocorrelation r of the returns.",
    "min_net_sharpe": "Statistical: the validation net Sharpe must be above this.", "min_sharpe_lower_bound": "Statistical: the lower end of the HAC interval of the Sharpe ratio must be above this (off by default).",
    "sharpe_level": "The confidence level of that interval.",
    "max_label_overlap": "Statistical: the number of walk-forward folds whose training labels reached past the date their fit may inform, read from the lineage the experiment recorded. Any is a leak; an experiment with no recorded lineage is unavailable.", "missing_evidence": "What an enforced gate does when its evidence was not produced: ineligible, or skip (and still reported as unavailable).",
    "rule": "one_se is a selection heuristic (gap <= k standard errors of the paired gap); non_inferiority is a formal one-sided test against a margin you declare.",
    "k": "one_se: the gap to the highest-ranked candidate, in standard errors of the paired gap, that still counts as 'not distinguishable'.",
    "margin": "non_inferiority: the largest loss of the primary metric you accept, in its own units, declared before the results.", "level": "non_inferiority: the one-sided level, Holm-adjusted over the comparisons.",
    "priority": "The order in which complexity dimensions decide: model_complexity, n_features, dependencies, training_cost, inference_latency, stability.",
    "cost_tolerance": "Relative difference under which two training costs, latencies or instabilities are a tie.", "n_boot": "Stationary-bootstrap replications of the paired gap.",
    "require_paired": "Refuse an unpaired standard error for the gap between two candidates.", "enabled": "Switch the simplicity step off to keep the highest-ranked candidate.",
    "nested": "The nested estimate of the ranking rule: what choosing by the primary metric on the days known at each cutoff earned on the days after it (the net Sharpe ranking only).",
    "cutoff_fractions": "Where the cutoffs fall, as shares of the pooled validation days.", "inner_days": "The most recent days before a cutoff the ranking rule looks at.",
    "min_inner": "Fewer inner days than this and a cutoff makes no choice.",
    "seed": "The seed of the bootstrap that estimates the paired gap, so a selection can be repeated exactly.",
    "pareto": "The objectives of the Pareto comparison; by default the primary metric and the secondary criteria.",
    "blocks": "How many consecutive chronological blocks of the validation segment the stability (instability of the block Sharpe ratios) is measured over.",
    "min_block_days": "A block shorter than this is left out of the stability figure.",
}


class SelectionRoutes:
    def selection_get_routes(self) -> dict:
        return {"/api/ml/selection-policy": self.selection_policy, "/api/ml/selection": self.selection_read}

    def selection_post_routes(self) -> dict:
        return {"/api/ml/select": self.select_models}

    # ------------------------------------------------------------------------------------------------------------ the documented defaults
    def selection_policy(self, q: dict) -> dict:
        """``GET /api/ml/selection-policy``: the defaults of every threshold, with what each one does and the metrics and rules on offer."""
        from . import ml_api

        ml_api._only(q, set(), "/api/ml/selection-policy")
        pol = ms.SelectionPolicy()
        data = {"policy": asdict(pol), "text": POLICY_TEXT, "primary_metrics": list(ms.PRIMARY), "secondary": list(ms.SECONDARY), "pareto_objectives": list(ms.DIRECTION), "simplicity_rules": list(sm.RULES),
                "dimensions": list(sm.DIMENSIONS), "states": list(ms.STATES), "gate_families": ["data", "risk", "exposure", "liquidity", "statistical"],
                "note": "Every threshold is configurable. A candidate is ranked only if it passes the requirements; the validation segment is the only one read; the selection promotes nothing."}
        return clean(common.envelope("ml_selection_policy", data, {"registry": "platform"}, [], None, []))

    # ------------------------------------------------------------------------------------------------------------ read a stored selection
    def selection_read(self, q: dict) -> dict:
        """``GET /api/ml/selection?id=<selection id>``: a stored selection, exactly as it was recorded. Without ``id`` it lists the latest ones."""
        from . import ml_api

        ml_api._only(q, {"id", "limit"}, "/api/ml/selection")
        with ml_api._user_errors():
            if q.get("id"):
                sid = str(q["id"])
                if not sid.startswith("msel-") or not ml_api.ID.match(sid):
                    raise ApiError("id is a selection id such as msel-0123456789abcd")
                found = ms.stored_selection(self.registry, sid)
                if found is None:
                    raise ApiError(f"no selection '{sid}' is stored")
                return clean(common.envelope("ml_selection", found, {"registry": "platform"}, [], None, []))
            limit = ml_api._int(q, "limit", 20, 1, 200)
            with self.registry._conn() as c:                                                     # noqa: SLF001
                rows = c.execute("SELECT subject_id, MAX(seq) AS seq FROM analysis_records WHERE kind='model_selection' GROUP BY subject_id ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for r in rows:
                rec = dx.latest(self.registry, r["subject_id"], "model_selection")["content"]
                out.append({"selection_id": r["subject_id"], "record_seq": r["seq"], "selected": rec.get("selected"), "highest_ranked": rec.get("highest_ranked"), "n_candidates": rec.get("n_candidates"),
                            "counts": rec.get("counts"), "policy_hash": rec.get("policy_hash"), "primary_metric": rec.get("primary_metric"), "source": (rec.get("source") or {}).get("kind")})
        return clean(common.envelope("ml_selections", {"selections": out}, {"registry": "platform"}, [], None, []))

    # ------------------------------------------------------------------------------------------------------------ choose among saved experiments
    def select_models(self, body: dict) -> dict:
        """``POST /api/ml/select {"experiments": [ids], "policy": {...}, "until": "2024-12-31", "store": true, "actor": "name", "plan": "plan id", "audit": false}``: the constrained selection over saved experiments, on their validation rows
        only (test and forward rows of a finalised experiment are counted and never read). ``audit`` re-derives each fold's facts (the rows' hash and count, the last day any training label reaches, disjoint and ordered train and test rows) from the rows it recorded, which loads the saved panel. ``policy`` takes any field of ``/api/ml/selection-policy``; unknown ones are refused."""
        from . import ml_api

        ml_api._only(body, {"experiments", "policy", "until", "store", "actor", "plan", "audit"}, "/api/ml/select")
        with ml_api._user_errors():
            ids = ml_api._ids(body, "experiments", 1, 20)
            try:
                policy = ms.policy_from(body.get("policy"))
            except (ms.SelectionError, sm.SimplicityError) as exc:
                raise ApiError(str(exc)) from None
            until = ml_api._date("until", body["until"]) if body.get("until") else None
            for i in ids:
                self.registry.get_experiment(i)
            actor = str(body.get("actor") or "api").strip()[:80] or "api"
            with self._heavy:
                try:
                    out = ms.select_saved(self.registry, ids, policy, until=until, actor=actor, store=bool(body.get("store", True)), plan_id=str(body["plan"]) if body.get("plan") else None, audit=bool(body.get("audit", False)))
                except ms.SelectionError as exc:
                    raise ApiError(str(exc)) from None
        return clean(common.envelope("ml_selection", out, {"registry": "platform"}, [], None,
                                     ["Only the validation segment was read; the selection promotes nothing. A heuristic choice is not a test of equivalence: see claims."]))
