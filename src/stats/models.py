"""Model loading and in-process reuse for statistics workflows."""

from pathlib import Path
from contextlib import contextmanager


def _load_classifier(model_dir):
    """Load a sequence classifier with the attention backend used by this project."""
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=2, attn_implementation='eager',
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model


class ModelCache:
    """Load each selected model at most once within a process."""

    def __init__(self):
        self._models = {}

    def get(self, model_dir):
        source = str(model_dir)
        expanded = Path(model_dir).expanduser()
        key = str(expanded.resolve()) if expanded.exists() else source
        if key not in self._models:
            self._models[key] = _load_classifier(source)
        return self._models[key]

    def clear(self):
        """Release cached models and free unused CUDA allocations."""
        import torch

        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_ACTIVE_CACHE = None


@contextmanager
def use_model_cache(cache):
    """Use the supplied model cache for model loads in the current workflow."""
    global _ACTIVE_CACHE
    previous_cache = _ACTIVE_CACHE
    _ACTIVE_CACHE = cache
    try:
        yield cache
    finally:
        _ACTIVE_CACHE = previous_cache


def load_classifier(model_dir):
    """Load a model, reusing it when called inside ``use_model_cache``."""
    if _ACTIVE_CACHE is not None:
        return _ACTIVE_CACHE.get(model_dir)
    return _load_classifier(model_dir)
