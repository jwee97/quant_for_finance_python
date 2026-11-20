"""The research workspaces' endpoints: Research Evidence, the feature registry and its evidence, model-and-baseline comparison, and the scheduled retraining simulation.

``GET /api/ml/evidence?id=`` reads what a saved run recorded and applies stated rules (``src/ml/evidence.py``); ``GET /api/ml/features`` is the registry; the three jobs (``/api/ml/features/study``,
``/api/ml/compare-strategies``, ``/api/ml/simulate``) take the same experiment settings as ``/api/ml/train`` and run through ``src.ml``. Nothing here decides anything for the person.
"""

from __future__ import annotations

import json

from ..assets import common
from ..ml import evidence as evd
from ..ml import feature_registry as fr
from ..ml import simulation as sim
from ..ml import strategy_compare as scmp
from ..ml.experiments import prepare
from .api import ApiError
from .payload import clean


def _h():
    from . import ml_api

    return ml_api


GRIDS = {"default": None, "narrow": [{"id": "entry 20%, no stop", "policy": {"entry_quantile": 0.2, "exit_quantile": 0.4, "stop_loss_vol": None}},
                                    {"id": "entry 10%, no stop", "policy": {"entry_quantile": 0.1, "exit_quantile": 0.2, "stop_loss_vol": None}}]}


class ResearchRoutes:
    def research_get_routes(self) -> dict:
        return {"/api/ml/evidence": self.evidence, "/api/ml/features": self.features}

    def research_post_routes(self) -> dict:
        return {"/api/ml/features/study": self.features_study, "/api/ml/compare-strategies": self.compare_strategies, "/api/ml/simulate": self.simulate}

    # ------------------------------------------------------------------------------------------------------------ evidence
    def evidence(self, q: dict) -> dict:
        """``GET /api/ml/evidence?id=``: the checklist for one saved experiment, each line with its rule and the place its number came from."""
        h = _h()
        h._only(q, {"id"}, "/api/ml/evidence")
        with h._user_errors():
            exp_id = h._id(q)
            report = self.registry.report(exp_id)
            f = self.registry.experiment_dir(exp_id) / "controls.json"
            controls = json.loads(f.read_text()) if f.exists() else None
        return clean(common.envelope("ml_evidence", evd.build(report, controls), {"registry": "platform"}, [], None, []))

    # ------------------------------------------------------------------------------------------------------------ features
    def features(self, q: dict) -> dict:
        """``GET /api/ml/features[?group=]``: every feature with its source, when it is known, how it is calculated and the hypothesis and reference that make it worth testing."""
        h = _h()
        h._only(q, {"group"}, "/api/ml/features")
        group = q.get("group") or None
        if group and group not in fr.GROUPS:
            raise ApiError(f"group must be one of {list(fr.GROUPS)}")
        return clean(common.envelope("ml_features", {"groups": fr.groups_summary(), "features": fr.registry([group] if group else None), "version": fr.VERSION}, {"registry": "platform"}, [],
                                     None, ["A hypothesis is a reason to test a feature, not a result. The study measures each one on a dataset."]))

    def features_study(self, body: dict) -> dict:
        """``POST /api/ml/features/study`` (a job): the experiment settings of ``/api/ml/train`` plus ``ablation`` (bool) and ``ablate_groups`` (a list). Returns each feature's rank IC and stability and, if asked,
        the group ablations against a noise control."""
        h = _h()
        extra = {"ablation", "ablate_groups"}
        over = self._overrides({k: v for k, v in body.items() if k not in extra})
        do_ablate = bool(body.get("ablation", False))
        groups = body.get("ablate_groups")
        if groups is not None and (not isinstance(groups, list) or any(g not in fr.GROUPS for g in groups)):
            raise ApiError(f"ablate_groups must be a list drawn from {list(fr.GROUPS)}")

        def work(progress):
            from ..ml.config import experiment_config

            with h._user_errors():
                cfg = experiment_config(over)
                ds, info = prepare(cfg, None, self.app.config, self.app.builder, None, None, progress)
                progress("measuring each feature against the target")
                out = {"features": fr.feature_table(ds), "registry": {r["name"]: r for r in fr.registry()}, "dataset": {"n_features": len(ds.features), "target": ds.target, "n_dates": len(ds.dates)}}
                if do_ablate:
                    out["ablation"] = fr.ablation(ds, cfg, info, groups, progress=progress)
            return clean(out)

        return self._submit("Feature study", work)

    # ------------------------------------------------------------------------------------------------------------ comparison
    def compare_strategies(self, body: dict) -> dict:
        """``POST /api/ml/compare-strategies`` (a job): ``{models: [{kind, params?}, ...], draws?, ...}`` plus the experiment settings of ``/api/ml/train`` (position rule, costs, splits). Trains each model on the same
        dataset and splits, trades them identically and tabulates them with the baselines over their common days."""
        h = _h()
        models = body.get("models")
        if not isinstance(models, list) or not 1 <= len(models) <= scmp.MAX_MODELS:
            raise ApiError(f"models is a list of 1 to {scmp.MAX_MODELS} {{kind, params}} objects")
        cleaned = []
        for m in models:
            if not isinstance(m, dict):
                raise ApiError("each model is an object with a kind")
            over_m = self._overrides({"model": m})
            cleaned.append(over_m["model"])
        draws = int(h._num("draws", body.get("draws", 10), 2, 40, True))
        over = self._overrides({k: v for k, v in body.items() if k not in ("models", "draws", "model")})

        def work(progress):
            with h._user_errors():
                out = scmp.compare(cleaned, over, config=self.app.config, builder=self.app.builder, progress=progress, draws=draws)
            return clean(out)

        return self._submit("Model comparison", work)

    # ------------------------------------------------------------------------------------------------------------ simulation
    def simulate(self, body: dict) -> dict:
        """``POST /api/ml/simulate`` (a job): the scheduled walk-forward simulation. The experiment settings of ``/api/ml/train`` plus ``grid`` (``default``, ``narrow`` or a list of ``{id, policy}``). Returns the saved
        experiment's id and the simulation block: the model versions, the policy chosen at each refit and what it was chosen on, and the scheduled result beside each fixed policy."""
        h = _h()
        grid_in = body.get("grid", "default")
        if isinstance(grid_in, str):
            if grid_in not in GRIDS:
                raise ApiError(f"grid must be one of {list(GRIDS)} or a list of {{id, policy}}")
            grid = GRIDS[grid_in]
        elif isinstance(grid_in, list) and all(isinstance(g, dict) and "id" in g and isinstance(g.get("policy"), dict) for g in grid_in):
            grid = grid_in
        else:
            raise ApiError("grid is default, narrow or a list of {id, policy} objects")
        lookback = int(h._num("lookback_days", body.get("lookback_days", 252), 63, 1000, True))
        over = self._overrides({k: v for k, v in body.items() if k not in ("grid", "lookback_days")})

        def work(progress):
            with h._user_errors():
                try:
                    exp, block = sim.simulate(over, grid, config=self.app.config, builder=self.app.builder, registry=self.registry, lookback_days=lookback, progress=progress)
                except ValueError as exc:
                    raise ApiError(str(exc)) from None
            return clean({"id": exp.id, "report": exp.report, "simulation": block})

        return self._submit("Scheduled simulation", work)
