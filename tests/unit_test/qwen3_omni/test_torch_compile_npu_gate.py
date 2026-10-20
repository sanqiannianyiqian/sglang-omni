# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni always disables torch.compile on NPU."""

from __future__ import annotations

from unittest.mock import patch

from sglang_omni.models.qwen3_omni.stages import _disable_torch_compile_on_npu
from sglang_omni.platforms import current_platform


def test_torch_compile_is_disabled_on_npu_when_explicitly_requested() -> None:
    with patch.object(type(current_platform), "is_npu", return_value=True):
        overrides = _disable_torch_compile_on_npu({"enable_torch_compile": True})

    assert overrides == {"enable_torch_compile": False}


def test_torch_compile_is_disabled_on_npu_when_unset() -> None:
    with patch.object(type(current_platform), "is_npu", return_value=True):
        overrides = _disable_torch_compile_on_npu(None)

    assert overrides == {"enable_torch_compile": False}


def test_torch_compile_is_unchanged_off_npu() -> None:
    with patch.object(type(current_platform), "is_npu", return_value=False):
        overrides = _disable_torch_compile_on_npu({"enable_torch_compile": True})

    assert overrides == {"enable_torch_compile": True}
