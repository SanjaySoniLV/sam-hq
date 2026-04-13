import argparse
import os
import time
import urllib.request
from typing import Any, Callable, Sequence, Tuple

import numpy as np
import onnxruntime
import torch
from torch import nn
from torchvision.io import ImageReadMode, read_image

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel


DEFAULT_CHECKPOINT_URLS = {
    "vit_tiny": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth",
    "vit_b": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth",
}
DEFAULT_DOG_IMAGE_PROMPT_POINTS = (
    (0.52, 0.56),  # Positive point near the subject center in demo/input_imgs/dog.jpg.
    (0.70, 0.78),  # Negative point near background to improve mask disambiguation.
)
DEFAULT_PROMPT_LABELS = (1, 0)  # 1=foreground point, 0=background point.


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
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Downloading checkpoint to {checkpoint_path} ...")
    urllib.request.urlretrieve(checkpoint_url, checkpoint_path)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _load_rgb_image(image_path: str) -> np.ndarray:
    try:
        image = read_image(image_path, mode=ImageReadMode.RGB)
    except (RuntimeError, OSError, FileNotFoundError) as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to read image at '{image_path}'. Provide a valid RGB JPEG/PNG image path."
        ) from exc
    return image.permute(1, 2, 0).cpu().numpy()


def _build_parity_inputs(sam, image: np.ndarray):
    """Build encoder/decoder inputs and predictor reference outputs on a real image."""
    predictor = SamPredictor(sam)

    # Build transformed_image explicitly so the same tensor can be used as encoder ONNX input.
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
    point_coords_unscaled = np.array(
        [[w * x_rel, h * y_rel] for x_rel, y_rel in DEFAULT_DOG_IMAGE_PROMPT_POINTS],
        dtype=np.float32,
    )
    point_labels_np = np.array(DEFAULT_PROMPT_LABELS, dtype=np.int64)
    point_coords_scaled = predictor.transform.apply_coords(point_coords_unscaled, image.shape[:2])
    point_coords = torch.as_tensor(point_coords_scaled, dtype=torch.float32, device=sam.device)[None, :, :]
    point_labels = torch.as_tensor(point_labels_np, dtype=torch.float32, device=sam.device)[None, :]
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

    with torch.no_grad():
        predictor_outputs = predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels.to(torch.int64),
            boxes=None,
            mask_input=None,
            multimask_output=False,
            return_logits=True,
            hq_token_only=False,
        )

    return {
        "encoder_input_image": input_image_torch,
        "decoder_inputs": decoder_inputs,
        "predictor_outputs": predictor_outputs,
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


def _benchmark(label: str, fn: Callable[[], Any], warmup: int, runs: int) -> float:
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
    model_type: str,
    image_path: str,
    checkpoint_path: str,
    decoder_output: str,
    encoder_output: str,
    checkpoint_url: str | None,
    opset: int,
    atol: float,
    rtol: float,
    benchmark_warmup: int,
    benchmark_runs: int,
) -> None:
    if model_type not in DEFAULT_CHECKPOINT_URLS:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Expected one of {tuple(DEFAULT_CHECKPOINT_URLS.keys())}."
        )
    checkpoint_url = checkpoint_url or DEFAULT_CHECKPOINT_URLS[model_type]
    _download_if_needed(checkpoint_path, checkpoint_url)

    print(f"Loading {model_type} model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.eval()

    decoder_model = SamOnnxModel(model=sam, hq_token_only=False, multimask_output=False)
    decoder_model.eval()

    encoder_model = SamTinyImageEncoderOnnxModel(sam)
    encoder_model.eval()

    image = _load_rgb_image(image_path)
    parity_data = _build_parity_inputs(sam, image)
    encoder_input_image = parity_data["encoder_input_image"]
    decoder_inputs = parity_data["decoder_inputs"]
    predictor_outputs = parity_data["predictor_outputs"]

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
        pt_outputs=predictor_outputs,
        ort_outputs=ort_pipeline_outputs,
        atol=atol,
        rtol=rtol,
        prefix="quality.predictor_vs_onnx_pipeline",
    )
    _check_outputs_close(
        names=["masks", "iou_predictions", "low_res_masks"],
        pt_outputs=pt_decoder_outputs,
        ort_outputs=ort_pipeline_outputs,
        atol=atol,
        rtol=rtol,
        prefix="pipeline",
    )

    def _pt_encoder_run():
        with torch.no_grad():
            encoder_model(encoder_input_image)

    encoder_pt_ms = _benchmark(
        "encoder.pytorch",
        _pt_encoder_run,
        warmup=benchmark_warmup,
        runs=benchmark_runs,
    )
    encoder_ort_ms = _benchmark(
        "encoder.onnxruntime",
        lambda: encoder_ort.run(None, encoder_ort_inputs),
        warmup=benchmark_warmup,
        runs=benchmark_runs,
    )

    def _pt_decoder_run():
        with torch.no_grad():
            decoder_model(**decoder_inputs)

    decoder_pt_ms = _benchmark(
        "decoder.pytorch",
        _pt_decoder_run,
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
        with torch.no_grad():
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
    print(f"Success: {model_type} encoder+decoder ONNX export completed and validated.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export HQ-SAM image encoder + mask decoder to ONNX, validate parity "
            "against ONNXRuntime and SamPredictor on a real JPEG, and print performance."
        )
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_tiny",
        choices=list(DEFAULT_CHECKPOINT_URLS.keys()),
        help="HQ-SAM model type to export.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. If missing, a /tmp path based on --model-type is used.",
    )
    parser.add_argument(
        "--checkpoint-url",
        type=str,
        default=None,
        help="Checkpoint URL used when --checkpoint does not exist.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="demo/input_imgs/dog.jpg",
        help="Path to the JPEG image used for end-to-end parity validation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path for the mask decoder.",
    )
    parser.add_argument(
        "--encoder-output",
        type=str,
        default=None,
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
    checkpoint_path = args.checkpoint or f"/tmp/sam-hq-{args.model_type}/sam_hq_{args.model_type}.pth"
    decoder_output = args.output or f"/tmp/sam-hq-{args.model_type}/sam_hq_{args.model_type}_decoder.onnx"
    encoder_output = (
        args.encoder_output or f"/tmp/sam-hq-{args.model_type}/sam_hq_{args.model_type}_encoder.onnx"
    )

    export_and_validate(
        model_type=args.model_type,
        image_path=args.image,
        checkpoint_path=checkpoint_path,
        decoder_output=decoder_output,
        encoder_output=encoder_output,
        checkpoint_url=args.checkpoint_url,
        opset=args.opset,
        atol=args.atol,
        rtol=args.rtol,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_runs=args.benchmark_runs,
    )


if __name__ == "__main__":
    main()
