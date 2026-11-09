"""Device selection shared by instruments. A request is not a usage claim."""
import os


def select_device(override=None):
    requested = override if override is not None else os.environ.get("GGTTS_INSTRUMENT_DEVICE", "cpu")
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda or auto")
    if requested == "cpu":
        return requested, "cpu"
    try:
        import torch
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested for instrument but unavailable")
    return requested, "cuda" if available else "cpu"


def verify_model_device(model, device):
    """For CUDA, refuse a model unless its parameters are all demonstrably on CUDA."""
    if device != "cuda":
        return
    candidates = [model, getattr(model, "model", None)]
    for candidate in candidates:
        parameters = getattr(candidate, "parameters", None)
        if not callable(parameters):
            continue
        try:
            devices = [getattr(getattr(p, "device", None), "type", None) for p in parameters()]
            if devices and all(d == "cuda" for d in devices):
                return
        except (StopIteration, TypeError):
            pass
    raise RuntimeError("instrument model did not demonstrate CUDA placement")

