__version__ = "0.1.0"

# Apply internal monkey-patches (e.g. BioPandas fixes) on import
try:
    from .biopandas_patch import apply_patch as _bp_apply_patch
    _bp_apply_patch()
except Exception as _e:  # pragma: no cover – patch failures should never crash
    # Log or silently ignore; we don't want import errors due to optional deps
    import warnings
    warnings.warn(f"BioPandas patch failed: {_e}")
finally:
    # Clean up namespace
    del _bp_apply_patch