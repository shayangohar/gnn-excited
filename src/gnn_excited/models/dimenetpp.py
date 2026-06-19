from __future__ import annotations

try:
    from torch_geometric.nn.models import DimeNetPlusPlus
except ModuleNotFoundError as exc:  # pragma: no cover
    DimeNetPlusPlus = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def build_dimenetpp(**kwargs):
    """Build a PyG DimeNet++ model for two scalar outputs."""
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "DimeNet++ requires torch_geometric. Install the ML environment before training."
        ) from _IMPORT_ERROR
    return DimeNetPlusPlus(**kwargs)
