"""Research operations: the infrastructure around the models.

    store       content-addressed ArtifactStore (hash-verified) and a versioned ModelRegistry with stages and lineage
    cv          purged k-fold, embargo, combinatorial purged cross-validation (CPCV) with path reassembly
    hpo         random / grid / TPE / successive-halving search that counts its trials and reports deflated Sharpe, PBO and Romano-Wolf for the winner; tune_pipeline
    profiling   profile_call, benchmark_models, scaling_exponent
"""

from . import cv, hpo, profiling, store
from .hpo import Categorical, Float, Int, LogFloat, Study, tune_pipeline
from .store import ArtifactStore, ModelRegistry
