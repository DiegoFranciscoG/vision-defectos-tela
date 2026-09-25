"""ONNX export, INT8 static quantization (QDQ, S8S8) and parity checks (#26, #27)."""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn

logger = logging.getLogger(__name__)


def export_onnx(
    model: nn.Module, sample: torch.Tensor, path: Path, output_names: list[str]
) -> Path:
    """Export with the default torch.export-based exporter and a dynamic batch dimension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    batch = torch.export.Dim("batch", min=1, max=64)
    program = torch.onnx.export(
        model,
        (sample,),
        dynamo=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_shapes=({0: batch},),
        external_data=False,
        verbose=False,
    )
    if program is None:
        raise RuntimeError("ONNX export returned no program")
    program.optimize()
    program.save(str(path), external_data=False)
    return path


class _Calibration(CalibrationDataReader):  # type: ignore[misc]
    def __init__(self, batches: list[np.ndarray]) -> None:
        self._iterator = iter([{"images": batch} for batch in batches])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iterator, None)


def quantize_int8(source: Path, target: Path, calibration: list[np.ndarray]) -> Path:
    """Quantize only convolutions: the nearest-neighbour distance and the Gaussian blur of
    PatchCore stay in float, because quantizing distances would destroy the score scale."""
    preprocessed = target.with_suffix(".pre.onnx")
    quant_pre_process(str(source), str(preprocessed), skip_symbolic_shape=True)
    quantize_static(
        str(preprocessed),
        str(target),
        _Calibration(calibration),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv"],
    )
    preprocessed.unlink(missing_ok=True)
    return target


def onnx_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_onnx(session: ort.InferenceSession, batch: np.ndarray) -> list[np.ndarray]:
    outputs: list[np.ndarray] = session.run(None, {"images": batch.astype(np.float32)})
    return outputs


def max_abs_difference(
    model: nn.Module, session: ort.InferenceSession, batch: torch.Tensor
) -> float:
    with torch.inference_mode():
        expected = model(batch)
    actual = run_onnx(session, batch.numpy())
    return max(
        float(np.max(np.abs(torch_out.numpy() - onnx_out)))
        for torch_out, onnx_out in zip(expected, actual, strict=True)
    )
