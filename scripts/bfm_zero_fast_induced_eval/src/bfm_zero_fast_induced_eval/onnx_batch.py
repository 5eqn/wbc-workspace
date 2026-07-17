from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_dynamic_batch(source: Path, cache_dir: Path) -> Path:
    import onnx

    source_hash = sha256_file(source)
    destination = cache_dir / source_hash / "FBcprAuxModel.dynamic-batch.onnx"
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    for value_info in [*model.graph.input, *model.graph.output]:
        dimension = value_info.type.tensor_type.shape.dim[0]
        dimension.ClearField("dim_value")
        dimension.dim_param = "batch"
    onnx.checker.check_model(model)
    temporary = destination.with_suffix(".tmp")
    onnx.save(model, temporary)
    temporary.replace(destination)
    return destination


def verify_dynamic_batches(
    source: Path, patched: Path, batch_sizes: tuple[int, ...] = (2, 128)
) -> dict[str, object]:
    import onnxruntime as ort

    ort.preload_dlls()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    original = ort.InferenceSession(str(source), providers=providers)
    dynamic = ort.InferenceSession(str(patched), providers=providers)
    if dynamic.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA ONNX Runtime is required, got {dynamic.get_providers()}")
    rng = np.random.default_rng(0)
    one = rng.normal(size=(1, 721)).astype(np.float32)
    reference = original.run(None, {original.get_inputs()[0].name: one})[0]
    parity = dynamic.run(None, {dynamic.get_inputs()[0].name: one})[0]
    np.testing.assert_allclose(parity, reference, rtol=1e-5, atol=1e-5)
    checks: dict[str, object] = {"batch_1_max_abs_error": float(np.max(np.abs(parity - reference)))}
    for batch_size in batch_sizes:
        tiled = np.repeat(one, batch_size, axis=0)
        output = dynamic.run(None, {dynamic.get_inputs()[0].name: tiled})[0]
        if output.shape != (batch_size, 29):
            raise ValueError(f"dynamic ONNX returned {output.shape} for batch {batch_size}")
        # CUDA GEMM kernels change with batch size; require close policy output, not bit parity.
        np.testing.assert_allclose(output, np.repeat(reference, batch_size, axis=0), rtol=1e-3, atol=1e-3)
        checks[f"batch_{batch_size}_shape"] = list(output.shape)
    checks["providers"] = dynamic.get_providers()
    return checks
