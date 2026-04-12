import argparse
import os
import urllib.request

import numpy as np
import onnxruntime
import torch

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel


DEFAULT_TINY_CHECKPOINT_URL = (
    "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth"
)


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
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    image_embeddings = predictor.features
    interm_embeddings = torch.stack(predictor.interm_features, dim=0)

    h, w = image.shape[:2]
    point_coords = torch.tensor([[[w / 2.0, h / 2.0]]], dtype=torch.float32, device=sam.device)
    point_labels = torch.tensor([[1.0]], dtype=torch.float32, device=sam.device)
    mask_input = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=sam.device)
    has_mask_input = torch.tensor([0.0], dtype=torch.float32, device=sam.device)
    orig_im_size = torch.tensor([float(h), float(w)], dtype=torch.float32, device=sam.device)

    return {
        "image_embeddings": image_embeddings,
        "interm_embeddings": interm_embeddings,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
        "orig_im_size": orig_im_size,
    }


def export_and_validate(
    checkpoint_path: str,
    output: str,
    checkpoint_url: str,
    opset: int,
    atol: float,
    rtol: float,
) -> None:
    _download_if_needed(checkpoint_path, checkpoint_url)

    print("Loading vit_tiny model...")
    sam = sam_model_registry["vit_tiny"](checkpoint=checkpoint_path)
    sam.eval()

    onnx_model = SamOnnxModel(model=sam, hq_token_only=False, multimask_output=False)
    onnx_model.eval()

    parity_inputs = _build_parity_inputs(sam, _make_dummy_circle_image())
    _ = onnx_model(**parity_inputs)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    print(f"Exporting ONNX to {output} ...")
    with open(output, "wb") as f:
        torch.onnx.export(
            onnx_model,
            tuple(parity_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=list(parity_inputs.keys()),
            output_names=["masks", "iou_predictions", "low_res_masks"],
            dynamic_axes=dynamic_axes,
        )

    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(output, providers=providers)
    ort_inputs = {k: _to_numpy(v) for k, v in parity_inputs.items()}

    with torch.no_grad():
        pt_outputs = onnx_model(**parity_inputs)
    ort_outputs = ort_session.run(None, ort_inputs)

    names = ["masks", "iou_predictions", "low_res_masks"]
    for idx, name in enumerate(names):
        pt = _to_numpy(pt_outputs[idx])
        ort = ort_outputs[idx]
        max_abs = float(np.max(np.abs(pt - ort)))
        is_close = np.allclose(pt, ort, atol=atol, rtol=rtol)
        print(f"{name}: max_abs_diff={max_abs:.8f}, allclose={is_close}")
        if not is_close:
            raise RuntimeError(
                f"{name} mismatch between PyTorch and ONNXRuntime. "
                f"max_abs_diff={max_abs:.8f}, atol={atol}, rtol={rtol}"
            )

    print("Success: vit_tiny ONNX export completed and outputs are near-identical.")


def main():
    parser = argparse.ArgumentParser(
        description="Export HQ-SAM vit_tiny to ONNX and validate parity against ONNXRuntime."
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
        default="/tmp/sam-hq-vit-tiny/sam_hq_vit_tiny.onnx",
        help="Output ONNX file path.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for parity.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for parity.")
    args = parser.parse_args()

    export_and_validate(
        checkpoint_path=args.checkpoint,
        output=args.output,
        checkpoint_url=args.checkpoint_url,
        opset=args.opset,
        atol=args.atol,
        rtol=args.rtol,
    )


if __name__ == "__main__":
    main()
