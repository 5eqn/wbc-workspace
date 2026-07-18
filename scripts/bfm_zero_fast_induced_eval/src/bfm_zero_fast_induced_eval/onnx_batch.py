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


def patch_dynamic_batch(
    source: Path, cache_dir: Path, input_dim: int | None = None, output_dim: int = 29
) -> Path:
    import onnx

    source_hash = sha256_file(source)
    destination = cache_dir / source_hash / f"{source.stem}.dynamic-batch.onnx"
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    actual_input = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    actual_output = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
    if input_dim is not None and actual_input != input_dim:
        raise ValueError(f"ONNX input dimension is {actual_input}, expected {input_dim}")
    if actual_output != output_dim:
        raise ValueError(f"ONNX output dimension is {actual_output}, expected {output_dim}")
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
    source: Path,
    patched: Path,
    batch_sizes: tuple[int, ...] = (2, 128),
    *,
    input_dim: int = 721,
    output_dim: int = 29,
    torchscript: Path | None = None,
) -> dict[str, object]:
    import onnxruntime as ort

    ort.preload_dlls()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    original = ort.InferenceSession(str(source), providers=providers)
    dynamic = ort.InferenceSession(str(patched), providers=providers)
    if dynamic.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA ONNX Runtime is required, got {dynamic.get_providers()}")
    rng = np.random.default_rng(0)
    one = rng.normal(size=(1, input_dim)).astype(np.float32)
    reference = original.run(None, {original.get_inputs()[0].name: one})[0]
    parity = dynamic.run(None, {dynamic.get_inputs()[0].name: one})[0]
    np.testing.assert_allclose(parity, reference, rtol=1e-5, atol=1e-5)
    checks: dict[str, object] = {"batch_1_max_abs_error": float(np.max(np.abs(parity - reference)))}
    for batch_size in batch_sizes:
        tiled = np.repeat(one, batch_size, axis=0)
        output = dynamic.run(None, {dynamic.get_inputs()[0].name: tiled})[0]
        if output.shape != (batch_size, output_dim):
            raise ValueError(f"dynamic ONNX returned {output.shape} for batch {batch_size}")
        # CUDA GEMM kernels change with batch size; require close policy output, not bit parity.
        np.testing.assert_allclose(output, np.repeat(reference, batch_size, axis=0), rtol=1e-3, atol=1e-3)
        checks[f"batch_{batch_size}_shape"] = list(output.shape)
    checks["providers"] = dynamic.get_providers()
    if torchscript is not None:
        import torch

        module = torch.jit.load(str(torchscript), map_location="cpu").eval()
        with torch.inference_mode():
            torch_output = module(torch.from_numpy(one)).detach().cpu().numpy()
        np.testing.assert_allclose(reference, torch_output, rtol=1e-5, atol=1e-5)
        checks["torchscript_batch_1_max_abs_error"] = float(np.max(np.abs(reference - torch_output)))
    return checks
