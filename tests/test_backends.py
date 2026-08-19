from __future__ import annotations

import pytest

from bg_removal.backends import BiRefNetBackend, _select_output_tensor


def test_select_output_tensor_uses_last_nested_prediction() -> None:
    marker = object()
    assert _select_output_tensor(["aux", ["aux-2", marker]]) is marker


def test_birefnet_gives_actionable_error_without_runtime_dependencies() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="pip install"):
            BiRefNetBackend()
    else:
        pytest.skip("Heavy runtime is installed; unit tests must not download weights")
