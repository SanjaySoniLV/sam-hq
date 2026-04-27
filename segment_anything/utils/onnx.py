# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Tuple

from ..modeling import Sam
from .amg import calculate_stability_score


class SamOnnxModel(nn.Module):
    """
    This model should not be called directly, but is used in ONNX export.
    It combines the prompt encoder, mask decoder, and mask postprocessing of Sam,
    with some functions modified to enable model tracing. Also supports extra
    options controlling what information. See the ONNX export script for details.
    """

    def __init__(
        self,
        model: Sam,
        hq_token_only: bool = False,
        multimask_output: bool = False,
        use_stability_score: bool = False,
        return_extra_metrics: bool = False,
        skip_mask_postprocessing: bool = False,
    ) -> None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.hq_token_only = hq_token_only
        self.multimask_output = multimask_output
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.return_extra_metrics = return_extra_metrics
        self.skip_mask_postprocessing = skip_mask_postprocessing

    @staticmethod
    def resize_longest_image_size(
        input_image_size: torch.Tensor, longest_side: int
    ) -> torch.Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        """Match `PromptEncoder` when `boxes is None` (padded last point) without indexed writes that break ONNX."""
        pl = point_labels.to(torch.float32)
        padding_point = torch.zeros((point_coords.shape[0], 1, 2), device=point_coords.device, dtype=point_coords.dtype)
        padding_label = torch.full((pl.shape[0], 1), -1.0, device=pl.device, dtype=pl.dtype)
        coords = torch.cat([point_coords, padding_point], dim=1)
        labels = torch.cat([pl, padding_label], dim=1)
        coords = coords + 0.5
        pe = self.model.prompt_encoder.pe_layer.forward_with_coords(
            coords, self.model.prompt_encoder.input_image_size
        )
        le = labels.unsqueeze(-1).expand_as(pe)
        not_w = self.model.prompt_encoder.not_a_point_embed.weight
        out = pe * (le != -1) + not_w * (le == -1)
        for i in range(self.model.prompt_encoder.num_point_embeddings):
            w = self.model.prompt_encoder.point_embeddings[i].weight
            out = out + w * (le == float(i))
        return out

    def _embed_masks(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor) -> torch.Tensor:
        if has_mask_input.dim() == 1:
            has_mask_input = has_mask_input.view(-1, 1, 1, 1)
        else:
            has_mask_input = has_mask_input.view(has_mask_input.shape[0], 1, 1, 1)
        mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(input_mask)
        no_mask = self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        mask_embedding = mask_embedding + (1 - has_mask_input) * no_mask
        return mask_embedding

    def mask_postprocessing(
        self,
        masks: torch.Tensor,
        orig_im_size: torch.Tensor,
        padded_im_size: torch.Tensor,
    ) -> torch.Tensor:
        """Match Sam.postprocess_masks: crop to `padded` (H,W) then upscale to `orig` (H,W).

        `padded_im_size` and `orig_im_size` are (B,2) in (H, W) order, matching SamPredictor.
        For a batch, we assume the same (H, W) pair for all elements (typical for batched same-shape images).
        """
        if orig_im_size.dim() == 1:
            oi = orig_im_size.unsqueeze(0)
        else:
            oi = orig_im_size
        if padded_im_size.dim() == 1:
            pad = padded_im_size.unsqueeze(0)
        else:
            pad = padded_im_size
        b = oi.shape[0]
        ph = pad[0, 0].to(torch.int64)
        pw = pad[0, 1].to(torch.int64)
        h0 = oi[0, 0].to(torch.int64)
        w0 = oi[0, 1].to(torch.int64)
        if b != 1 and (b != pad.shape[0] or b != oi.shape[0]):
            raise ValueError("orig_im_size and padded_im_size must match mask batch and each other")
        for i in range(1, b):
            if (pad[i, 0] != ph) or (pad[i, 1] != pw) or (oi[i, 0] != h0) or (oi[i, 1] != w0):
                raise NotImplementedError(
                    "Batched postprocessing for mixed (H, W) per item is not supported; "
                    "use the same size for every batch element or run one image at a time."
                )
        masks = F.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[:, :, :ph, :pw]  # type: ignore[index]
        return F.interpolate(masks, (h0, w0), mode="bilinear", align_corners=False)


    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        interm_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
        padded_im_size: torch.Tensor,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        if interm_embeddings.dim() == 5:
            # (num_layers, B, H, W, C); HQ uses the first intermediate ViT feature map
            vit_tok = interm_embeddings[0]
        elif interm_embeddings.dim() == 4:
            # Single layer, batched: (B, H, W, C)
            vit_tok = interm_embeddings
        else:
            raise ValueError(
                f"interm_embeddings must be 4D or 5D, got shape {tuple(interm_embeddings.shape)}"
            )
        vit_features = vit_tok.permute(0, 3, 1, 2)  # early-layer ViT feature
        hq_features = self.model.mask_decoder.embedding_encoder(image_embeddings) + self.model.mask_decoder.compress_vit_feat(vit_features)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            hq_features=hq_features,
        )

        if self.use_stability_score:
            scores = calculate_stability_score(
                masks, self.model.mask_threshold, self.stability_score_offset
            )

        if self.multimask_output:
            # mask with highest score
            mask_slice = slice(1,self.model.mask_decoder.num_mask_tokens-1)
            scores = scores[:, mask_slice]
            scores, max_iou_idx = torch.max(scores,dim=1)
            scores = scores.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            scores = scores[:,mask_slice]
            masks_sam = masks[:,mask_slice]

        masks_hq = masks[:,slice(self.model.mask_decoder.num_mask_tokens-1, self.model.mask_decoder.num_mask_tokens)]

        if self.hq_token_only:
            masks = masks_hq
        else:
            masks = masks_sam + masks_hq

        if self.skip_mask_postprocessing:
            upscaled_masks = masks
        else:
            upscaled_masks = self.mask_postprocessing(masks, orig_im_size, padded_im_size)

        if self.return_extra_metrics:
            stability_scores = calculate_stability_score(
                upscaled_masks, self.model.mask_threshold, self.stability_score_offset
            )
            areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks

        return upscaled_masks, scores, masks
