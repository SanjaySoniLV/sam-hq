import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence, Tuple

import numpy as np
import onnxruntime
import torch
from torch import nn
from torchvision.io import ImageReadMode, read_image

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    except FileNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(f"Image file not found at '{image_path}'.") from exc
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Could not open image at '{image_path}'. Check file permissions and path validity."
        ) from exc
    except RuntimeError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to decode image at '{image_path}'. Provide a valid RGB JPEG/PNG image path."
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
    interm_embeddings = torch.stack(predictor.interm_features, dim=0)  # (L, 1, H, W, C)

    h, w = image.shape[:2]
    _, _, ph, pw = input_image_torch.shape
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
    orig_im_size = torch.tensor([[float(h), float(w)]], dtype=torch.float32, device=sam.device)
    padded_im_size = torch.tensor([[float(ph), float(pw)]], dtype=torch.float32, device=sam.device)

    decoder_inputs = {
        "image_embeddings": image_embeddings,
        "interm_embeddings": interm_embeddings,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
        "orig_im_size": orig_im_size,
        "padded_im_size": padded_im_size,
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


def _decoder_inputs_with_batch(
    decoder_inputs: dict, batch_size: int
) -> dict:
    """Batch multiple prompt groups on a single image (matches SAM mask_decoder.repeat_interleave).

    `image_embeddings` and `interm_embeddings` stay batch-1; only point/mask and size fields tile.
    """
    if batch_size <= 1:
        return decoder_inputs
    out = dict(decoder_inputs)
    for k in (
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
    ):
        v = decoder_inputs[k]
        out[k] = v.expand(batch_size, *v.shape[1:]).contiguous()
    oi = decoder_inputs["orig_im_size"]
    pd = decoder_inputs["padded_im_size"]
    if oi.dim() == 1:
        out["orig_im_size"] = oi.unsqueeze(0).expand(batch_size, 2).contiguous()
    else:
        out["orig_im_size"] = oi.expand(batch_size, 2).contiguous()
    if pd.dim() == 1:
        out["padded_im_size"] = pd.unsqueeze(0).expand(batch_size, 2).contiguous()
    else:
        out["padded_im_size"] = pd.expand(batch_size, 2).contiguous()
    return out


def _decoder_inputs_slice_index(dec: dict, index: int, expected_batch: int) -> dict:
    out: dict = {}
    for k, v in dec.items():
        if k == "interm_embeddings" and v.dim() == 5 and v.shape[1] == expected_batch:
            out[k] = v[:, index : index + 1, ...]
        elif v.dim() >= 1 and v.shape[0] == expected_batch:
            out[k] = v[index : index + 1, ...]
        else:
            out[k] = v
    return out


def _patch_encoder_layernorm_inplace(encoder_onnx: Path) -> None:
    """Run LayerNorm decomp script in-place: backup original, write patched, replace."""
    orig = encoder_onnx.with_name(encoder_onnx.stem + ".unpatched" + encoder_onnx.suffix)
    shutil.copy2(encoder_onnx, orig)
    tmp_out = encoder_onnx.with_name(encoder_onnx.stem + ".ln_decomp_tmp" + encoder_onnx.suffix)
    script = REPO_ROOT / "scripts" / "rewrite_encoder_layernorm_to_primitive_ops.py"
    r = subprocess.run(
        [sys.executable, str(script), str(orig), str(tmp_out)],
        check=False,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            "LayerNorm rewrite failed.\n"
            f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )
    shutil.move(str(tmp_out), str(encoder_onnx))
    print(f"Encoder LayerNorm decomposed in-place: {encoder_onnx} (backup: {orig.name})")


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
    providers: Sequence[str],
    skip_mask_postprocessing: bool,
    interm_embeddings_stacked: bool,
    decompose_encoder_layernorm: bool,
    validate_batch: int,
) -> None:
    checkpoint_url = checkpoint_url or DEFAULT_CHECKPOINT_URLS[model_type]
    _download_if_needed(checkpoint_path, checkpoint_url)

    print(f"Loading {model_type} model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.eval()

    decoder_model = SamOnnxModel(
        model=sam,
        hq_token_only=False,
        multimask_output=False,
        skip_mask_postprocessing=skip_mask_postprocessing,
    )
    decoder_model.eval()

    encoder_model = SamTinyImageEncoderOnnxModel(sam)
    encoder_model.eval()

    image = _load_rgb_image(image_path)
    if os.path.basename(image_path) != "dog.jpg":
        print(
            "Warning: default validation prompts are tuned for demo/input_imgs/dog.jpg; "
            "custom images may require different prompt points."
        )
    parity_data = _build_parity_inputs(sam, image)
    encoder_input_image = parity_data["encoder_input_image"]
    decoder_inputs = parity_data["decoder_inputs"]
    predictor_outputs = parity_data["predictor_outputs"]
    if not interm_embeddings_stacked:
        decoder_inputs["interm_embeddings"] = decoder_inputs["interm_embeddings"][0]

    _ = encoder_model(encoder_input_image)
    _ = decoder_model(**decoder_inputs)

    os.makedirs(os.path.dirname(decoder_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(encoder_output) or ".", exist_ok=True)

    print(f"Exporting encoder ONNX to {encoder_output} ...")
    encoder_dynamic = {"input_image": {0: "batch"}}
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
            dynamic_axes=encoder_dynamic,
            dynamo=False,
        )
    if decompose_encoder_layernorm:
        _patch_encoder_layernorm_inplace(Path(encoder_output))

    if interm_embeddings_stacked:
        # Encoder output is (L,1,H,W,C); first dim is layer, second is always 1 for this export path
        interm_dec_dyn = {0: "interm_layer"}
    else:
        interm_dec_dyn = {0: "batch"}
    dynamic_axes: dict = {
        "point_coords": {0: "num_prompt_groups", 1: "num_points"},
        "point_labels": {0: "num_prompt_groups", 1: "num_points"},
        "mask_input": {0: "num_prompt_groups"},
        "has_mask_input": {0: "num_prompt_groups"},
        "orig_im_size": {0: "num_prompt_groups"},
        "padded_im_size": {0: "num_prompt_groups"},
    }
    if not interm_embeddings_stacked:
        dynamic_axes["interm_embeddings"] = interm_dec_dyn

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

    available_providers = onnxruntime.get_available_providers()
    missing_providers = [provider for provider in providers if provider not in available_providers]
    if missing_providers:
        raise RuntimeError(
            f"Requested ONNX Runtime providers are unavailable: {missing_providers}. "
            f"Available providers: {available_providers}"
        )
    print(f"Using ONNX Runtime providers: {providers}")
    encoder_ort = onnxruntime.InferenceSession(encoder_output, providers=providers)
    decoder_ort = onnxruntime.InferenceSession(decoder_output, providers=providers)

    encoder_ort_inputs = {"input_image": _to_numpy(encoder_input_image)}
    decoder_ort_inputs = {k: _to_numpy(v) for k, v in decoder_inputs.items()}

    if decompose_encoder_layernorm:
        unpatched_path = (
            Path(encoder_output).with_name(
                Path(encoder_output).stem + ".unpatched" + Path(encoder_output).suffix
            )
        )
        if unpatched_path.is_file():
            print(
                f"Comparing ORT encoder outputs: unpatched ({unpatched_path.name}) vs LayerNorm-decomposed (active)..."
            )
            ort_unpatched = onnxruntime.InferenceSession(
                str(unpatched_path), providers=providers
            )
            out_un = ort_unpatched.run(None, encoder_ort_inputs)
            out_p = encoder_ort.run(None, encoder_ort_inputs)
            for name, a, b in zip(
                ["image_embeddings", "interm_embeddings"], out_un, out_p
            ):
                max_abs = float(np.max(np.abs(a - b)))
                is_close = np.allclose(a, b, atol=1e-4, rtol=1e-4)
                print(
                    f"encoder.unpatched_vs_ln_decomp.{name}: max_abs_diff={max_abs:.8e}, allclose(1e-4)={is_close}"
                )
                if not is_close:
                    raise RuntimeError(
                        f"LayerNorm decomp changed encoder {name} beyond 1e-4 tolerance: max_abs={max_abs}"
                    )

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
    parity_reference_outputs = predictor_outputs
    if skip_mask_postprocessing:
        # In DML-safe mode the decoder emits low-res logits for both "masks" and "low_res_masks".
        _, predictor_iou_predictions, predictor_low_res_masks = predictor_outputs
        parity_reference_outputs = (
            predictor_low_res_masks,
            predictor_iou_predictions,
            predictor_low_res_masks,
        )
    _check_outputs_close(
        names=["masks", "iou_predictions", "low_res_masks"],
        pt_outputs=parity_reference_outputs,
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

    if validate_batch > 1:
        print(
            f"Validating {validate_batch} prompt groups on one image (decoder batch / "
            "num_prompt_groups axis)..."
        )
        dec_b = _decoder_inputs_with_batch(decoder_inputs, validate_batch)
        with torch.no_grad():
            pt_b = decoder_model(**dec_b)
        dec_ort_b = {k: _to_numpy(v) for k, v in dec_b.items()}
        ort_b = decoder_ort.run(None, dec_ort_b)
        _check_outputs_close(
            names=["masks", "iou_predictions", "low_res_masks"],
            pt_outputs=pt_b,
            ort_outputs=ort_b,
            atol=atol,
            rtol=rtol,
            prefix=f"decoder.batch{validate_batch}",
        )
        enc_out_b = encoder_ort.run(None, encoder_ort_inputs)
        dec_from_enc_b = dict(dec_ort_b)
        dec_from_enc_b["image_embeddings"] = enc_out_b[0]
        dec_from_enc_b["interm_embeddings"] = enc_out_b[1]
        ort_pipe_b = decoder_ort.run(None, dec_from_enc_b)
        _check_outputs_close(
            names=["masks", "iou_predictions", "low_res_masks"],
            pt_outputs=pt_b,
            ort_outputs=ort_pipe_b,
            atol=atol,
            rtol=rtol,
            prefix=f"pipeline.batch{validate_batch}",
        )
        for i in range(validate_batch):
            dec_i = _decoder_inputs_slice_index(dec_b, i, validate_batch)
            with torch.no_grad():
                pt_i = decoder_model(**dec_i)
            for j, name in enumerate(["masks", "iou_predictions", "low_res_masks"]):
                ai = _to_numpy(pt_i[j])
                bi = _to_numpy(pt_b[j][i : i + 1])
                max_abs = float(np.max(np.abs(ai - bi)))
                is_close = np.allclose(ai, bi, atol=atol, rtol=rtol)
                print(
                    f"batch_parity.slice[{i}].{name}: max_abs_diff={max_abs:.8f}, allclose={is_close}"
                )
                if not is_close:
                    raise RuntimeError(
                        f"Batched output batch index {i} does not match independent run for {name}. "
                        f"max_abs_diff={max_abs}, atol={atol}, rtol={rtol}"
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
                padded_im_size=decoder_inputs["padded_im_size"],
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
        help=(
            "Path to the JPEG image used for end-to-end parity validation. "
            "Default prompt points are tuned for demo/input_imgs/dog.jpg."
        ),
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
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="ONNX Runtime execution providers to use in order, e.g. DmlExecutionProvider CPUExecutionProvider.",
    )
    parser.add_argument(
        "--skip-mask-postprocessing",
        action="store_true",
        help=(
            "Skip high-resolution mask postprocessing inside the exported decoder and emit low-res mask logits "
            "for both 'masks' and 'low_res_masks'. Useful for runtimes where dynamic resize/slice ops are unstable."
        ),
    )
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
    parser.add_argument(
        "--interm-embeddings-stacked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, interm_embeddings has shape (L,B,H,W,C) matching the encoder export. "
            "If false, use (B,H,W,C) for a single intermediate map (legacy)."
        ),
    )
    parser.add_argument(
        "--decompose-encoder-layernorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rewrite LayerNormalization in the encoder ONNX to primitive ops (DirectML fix). "
            "Saves a .unpatched copy next to the encoder for diff validation."
        ),
    )
    parser.add_argument(
        "--validate-batch",
        type=int,
        default=2,
        help=(
            "If >1, validate that many parallel prompt groups on a single image (matches "
            "how the SAM decoder uses its batch / num_prompt_groups axis). Set to 1 to skip."
        ),
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
        providers=args.providers,
        skip_mask_postprocessing=args.skip_mask_postprocessing,
        interm_embeddings_stacked=args.interm_embeddings_stacked,
        decompose_encoder_layernorm=args.decompose_encoder_layernorm,
        validate_batch=args.validate_batch,
    )


if __name__ == "__main__":
    main()
