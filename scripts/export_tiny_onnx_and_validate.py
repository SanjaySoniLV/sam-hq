import argparse
import os
import time
import urllib.request
from typing import Callable, Sequence, Tuple

import numpy as np
import onnxruntime
import torch
from torch import nn

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel


DEFAULT_TINY_CHECKPOINT_URL = (
    "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth"
)


class SamTinyImageEncoderOnnxModel(nn.Module):
    """ONNX-exportable wrapper around SAM image preprocessing + image encoder."""

    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model

    @torch.no_grad()
    def forward(self, input_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a transformed BCHW float image into image and intermediate embeddings."""
        preprocessed = self.model.preprocess(input_image)
        image_embeddings, interm_embeddings = self.model.image_encoder(preprocessed)
        if isinstance(interm_embeddings, list):
            interm_embeddings = torch.stack(interm_embeddings, dim=0)
        return image_embeddings, interm_embeddings


def _download_if_needed(checkpoint_path: str, checkpoint_url: str) -> None:
    if os.path.exists(checkpoint_path):
        return
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    print(f"Downloading checkpoint to {checkpoint_path} ...")
    urllib.request.urlretrieve(checkpoint_url, checkpoint_path)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _make_dummy_circle_image(size: int = 512) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 4
    mask = ((xx - center) ** 2 + (yy - center) ** 2) <= radius**2
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[mask] = np.array([255, 255, 255], dtype=np.uint8)
    return image


def _build_parity_inputs(sam, image: np.ndarray):
    """Build encoder and decoder inputs from a dummy image and center-point prompt."""
    predictor = SamPredictor(sam)

    input_image = predictor.transform.apply_image(image)
    input_image_torch = torch.as_tensor(
        input_image,
        dtype=torch.float32,
        device=sam.device,
    ).permute(2, 0, 1).contiguous()[None, :, :, :]
    predictor.set_torch_image(input_image_torch, image.shape[:2])

    if predictor.features is None:
        raise RuntimeError("SamPredictor did not produce image embeddings after set_image.")
    if predictor.interm_features is None or len(predictor.interm_features) == 0:
        raise RuntimeError("SamPredictor did not produce intermediate embeddings after set_image.")

    image_embeddings = predictor.features
    interm_embeddings = torch.stack(predictor.interm_features, dim=0)

    h, w = image.shape[:2]
    point_coords = torch.tensor([[[w / 2.0, h / 2.0]]], dtype=torch.float32, device=sam.device)
    point_labels = torch.tensor([[1.0]], dtype=torch.float32, device=sam.device)
    mask_input = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=sam.device)
    has_mask_input = torch.tensor([0.0], dtype=torch.float32, device=sam.device)
    orig_im_size = torch.tensor([float(h), float(w)], dtype=torch.float32, device=sam.device)

    decoder_inputs = {
        "image_embeddings": image_embeddings,
        "interm_embeddings": interm_embeddings,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
        "orig_im_size": orig_im_size,
    }

    return {
        "encoder_input_image": input_image_torch,
        "decoder_inputs": decoder_inputs,
    }


def _check_outputs_close(
    names: Sequence[str],
    pt_outputs: Sequence[torch.Tensor],
    ort_outputs: Sequence[np.ndarray],
    atol: float,
    rtol: float,
    prefix: str,
) -> None:
    """Validate ONNXRuntime outputs against PyTorch outputs and raise on mismatch."""
    for idx, name in enumerate(names):
        pt = _to_numpy(pt_outputs[idx])
        ort = ort_outputs[idx]
        max_abs = float(np.max(np.abs(pt - ort)))
        is_close = np.allclose(pt, ort, atol=atol, rtol=rtol)
        print(f"{prefix}.{name}: max_abs_diff={max_abs:.8f}, allclose={is_close}")
        if not is_close:
            raise RuntimeError(
                f"{prefix}.{name} mismatch between PyTorch and ONNXRuntime. "
                f"max_abs_diff={max_abs:.8f}, atol={atol}, rtol={rtol}"
            )


def _benchmark(label: str, fn: Callable[[], None], warmup: int, runs: int) -> float:
    """Benchmark a callable and return average execution time in milliseconds."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed * 1000.0 / runs
    print(f"perf.{label}: avg_ms={avg_ms:.3f} (warmup={warmup}, runs={runs})")
    return avg_ms


def _safe_speedup(torch_ms: float, ort_ms: float) -> str:
    """Return torch/onnxruntime speedup as text, or 'n/a' when divisor is non-positive."""
    if ort_ms <= 0.0:
        return "n/a"
    return f"{torch_ms / ort_ms:.2f}x"


def export_and_validate(
    checkpoint_path: str,
    decoder_output: str,
    encoder_output: str,
    checkpoint_url: str,
    opset: int,
    atol: float,
    rtol: float,
    benchmark_warmup: int,
    benchmark_runs: int,
) -> None:
    _download_if_needed(checkpoint_path, checkpoint_url)

    print("Loading vit_tiny model...")
    sam = sam_model_registry["vit_tiny"](checkpoint=checkpoint_path)
    sam.eval()

    decoder_model = SamOnnxModel(model=sam, hq_token_only=False, multimask_output=False)
    decoder_model.eval()

    encoder_model = SamTinyImageEncoderOnnxModel(sam)
    encoder_model.eval()

    parity_data = _build_parity_inputs(sam, _make_dummy_circle_image())
    encoder_input_image = parity_data["encoder_input_image"]
    decoder_inputs = parity_data["decoder_inputs"]

    _ = encoder_model(encoder_input_image)
    _ = decoder_model(**decoder_inputs)

    os.makedirs(os.path.dirname(decoder_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(encoder_output) or ".", exist_ok=True)

    print(f"Exporting encoder ONNX to {encoder_output} ...")
    with open(encoder_output, "wb") as f:
        torch.onnx.export(
            encoder_model,
            (encoder_input_image,),
            f,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["image_embeddings", "interm_embeddings"],
            dynamo=False,
        )

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    print(f"Exporting decoder ONNX to {decoder_output} ...")
    with open(decoder_output, "wb") as f:
        torch.onnx.export(
            decoder_model,
            tuple(decoder_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=list(decoder_inputs.keys()),
            output_names=["masks", "iou_predictions", "low_res_masks"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    providers = ["CPUExecutionProvider"]
    encoder_ort = onnxruntime.InferenceSession(encoder_output, providers=providers)
    decoder_ort = onnxruntime.InferenceSession(decoder_output, providers=providers)

    encoder_ort_inputs = {"input_image": _to_numpy(encoder_input_image)}
    decoder_ort_inputs = {k: _to_numpy(v) for k, v in decoder_inputs.items()}

    with torch.no_grad():
        pt_encoder_outputs = encoder_model(encoder_input_image)
    ort_encoder_outputs = encoder_ort.run(None, encoder_ort_inputs)
    _check_outputs_close(
        names=["image_embeddings", "interm_embeddings"],
        pt_outputs=pt_encoder_outputs,
        ort_outputs=ort_encoder_outputs,
        atol=atol,
        rtol=rtol,
        prefix="encoder",
    )

    with torch.no_grad():
        pt_decoder_outputs = decoder_model(**decoder_inputs)
    ort_decoder_outputs = decoder_ort.run(None, decoder_ort_inputs)
    _check_outputs_close(
        names=["masks", "iou_predictions", "low_res_masks"],
        pt_outputs=pt_decoder_outputs,
        ort_outputs=ort_decoder_outputs,
        atol=atol,
        rtol=rtol,
        prefix="decoder",
    )

    decoder_inputs_from_ort_encoder = dict(decoder_ort_inputs)
    decoder_inputs_from_ort_encoder["image_embeddings"] = ort_encoder_outputs[0]
    decoder_inputs_from_ort_encoder["interm_embeddings"] = ort_encoder_outputs[1]
    ort_pipeline_outputs = decoder_ort.run(None, decoder_inputs_from_ort_encoder)
    _check_outputs_close(
        names=["masks", "iou_predictions", "low_res_masks"],
        pt_outputs=pt_decoder_outputs,
        ort_outputs=ort_pipeline_outputs,
        atol=atol,
        rtol=rtol,
        prefix="pipeline",
    )

    encoder_pt_ms = _benchmark(
        "encoder.pytorch",
        lambda: encoder_model(encoder_input_image),
        warmup=benchmark_warmup,
        runs=benchmark_runs,
    )
    encoder_ort_ms = _benchmark(
        "encoder.onnxruntime",
        lambda: encoder_ort.run(None, encoder_ort_inputs),
        warmup=benchmark_warmup,
        runs=benchmark_runs,
    )

    decoder_pt_ms = _benchmark(
        "decoder.pytorch",
        lambda: decoder_model(**decoder_inputs),
        warmup=benchmark_warmup,
        runs=benchmark_runs,
    )
    decoder_ort_ms = _benchmark(
        "decoder.onnxruntime",
        lambda: decoder_ort.run(None, decoder_ort_inputs),
        warmup=benchmark_warmup,
        runs=benchmark_runs,
    )

    def _pt_pipeline_run():
        image_embeddings, interm_embeddings = encoder_model(encoder_input_image)
        decoder_model(
            image_embeddings=image_embeddings,
            interm_embeddings=interm_embeddings,
            point_coords=decoder_inputs["point_coords"],
            point_labels=decoder_inputs["point_labels"],
            mask_input=decoder_inputs["mask_input"],
            has_mask_input=decoder_inputs["has_mask_input"],
            orig_im_size=decoder_inputs["orig_im_size"],
        )

    def _ort_pipeline_run():
        image_embeddings, interm_embeddings = encoder_ort.run(None, encoder_ort_inputs)
        ort_inputs = dict(decoder_ort_inputs)
        ort_inputs["image_embeddings"] = image_embeddings
        ort_inputs["interm_embeddings"] = interm_embeddings
        decoder_ort.run(None, ort_inputs)

    pipeline_pt_ms = _benchmark(
        "pipeline.pytorch",
        _pt_pipeline_run,
        warmup=benchmark_warmup,
        runs=benchmark_runs,
    )
    pipeline_ort_ms = _benchmark(
        "pipeline.onnxruntime",
        _ort_pipeline_run,
        warmup=benchmark_warmup,
        runs=benchmark_runs,
    )

    print("Performance summary (avg ms, lower is better):")
    print(
        "  encoder: "
        f"pytorch={encoder_pt_ms:.3f}, onnxruntime={encoder_ort_ms:.3f}, "
        f"speedup={_safe_speedup(encoder_pt_ms, encoder_ort_ms)}"
    )
    print(
        "  decoder: "
        f"pytorch={decoder_pt_ms:.3f}, onnxruntime={decoder_ort_ms:.3f}, "
        f"speedup={_safe_speedup(decoder_pt_ms, decoder_ort_ms)}"
    )
    print(
        "  pipeline: "
        f"pytorch={pipeline_pt_ms:.3f}, onnxruntime={pipeline_ort_ms:.3f}, "
        f"speedup={_safe_speedup(pipeline_pt_ms, pipeline_ort_ms)}"
    )
    print("Success: vit_tiny encoder+decoder ONNX export completed and validated.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export HQ-SAM vit_tiny image encoder + mask decoder to ONNX, "
            "validate parity with ONNXRuntime, and print performance comparison."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/tmp/sam-hq-vit-tiny/sam_hq_vit_tiny.pth",
        help="Path to the vit_tiny checkpoint. If missing, it will be downloaded.",
    )
    parser.add_argument(
        "--checkpoint-url",
        type=str,
        default=DEFAULT_TINY_CHECKPOINT_URL,
        help="Checkpoint URL used when --checkpoint does not exist.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/sam-hq-vit-tiny/sam_hq_vit_tiny_decoder.onnx",
        help="Output ONNX file path for the mask decoder.",
    )
    parser.add_argument(
        "--encoder-output",
        type=str,
        default="/tmp/sam-hq-vit-tiny/sam_hq_vit_tiny_encoder.onnx",
        help="Output ONNX file path for the image encoder.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for parity.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for parity.")
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=3,
        help="Number of warmup runs for performance comparison.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=20,
        help="Number of timed runs for performance comparison.",
    )
    args = parser.parse_args()

    export_and_validate(
        checkpoint_path=args.checkpoint,
        decoder_output=args.output,
        encoder_output=args.encoder_output,
        checkpoint_url=args.checkpoint_url,
        opset=args.opset,
        atol=args.atol,
        rtol=args.rtol,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_runs=args.benchmark_runs,
    )


if __name__ == "__main__":
    main()
