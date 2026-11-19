# A container shows the host's CPUs while a quota limits what it may use; numerical libraries that start one thread per host CPU then fight over the quota (see src/ml/resources.py).
# The limits are set here, before numpy and torch are imported by anything else, and never over a value the person set.
try:
    from .ml.resources import apply_thread_limits

    apply_thread_limits()
except Exception:                                        # nothing about the machine may stop the package from importing
    pass
