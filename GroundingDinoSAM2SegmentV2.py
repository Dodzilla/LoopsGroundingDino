
import os
import sys
import uuid
import copy
import numpy as np
import cv2
import torch
from PIL import Image

import folder_paths
import comfy.model_management

# External modules already present in the original node package
# Match node.py behavior: ensure this package dir is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sam2.sam2_image_predictor import SAM2ImagePredictor
from local_groundingdino.datasets import transforms as T

# -----------------------------
# Helper functions (V2 specific)
# -----------------------------

def _load_dino_image(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def _get_grounding_output_with_scores(model, image, caption, box_threshold):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = comfy.model_management.get_torch_device()
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    # filter by threshold on best score across text tokens
    scores = logits.max(dim=1)[0]  # (nq,)
    keep = scores > box_threshold
    return boxes[keep].cpu(), scores[keep].cpu()

def _xywhn_to_xyxy_abs(boxes, W, H):
    """
    convert cx,cy,w,h in [0,1] -> x1,y1,x2,y2 in pixels
    boxes: (N,4) tensor
    """
    out = boxes.clone()
    out *= torch.tensor([W, H, W, H])
    out[:, :2] -= out[:, 2:] / 2
    out[:, 2:] += out[:, :2]
    return out

def _nms_xyxy(boxes, scores, iou_thr=0.5):
    """
    Simple NMS for xyxy boxes.
    boxes: (N,4) tensor
    scores: (N,) tensor
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][ovr <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)

def groundingdino_predict_v2(dino_model, image_pil, prompt, threshold, max_bounding_boxes, nms_iou=0.5):
    """
    Returns top-K boxes AFTER score sorting + NMS.
    """
    dino_image = _load_dino_image(image_pil.convert("RGB"))
    boxes_n, scores = _get_grounding_output_with_scores(dino_model, dino_image, prompt, threshold)
    W, H = image_pil.size

    if boxes_n.numel() == 0:
        return torch.empty((0,4), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)

    boxes_xyxy = _xywhn_to_xyxy_abs(boxes_n, W, H)

    # NMS
    keep = _nms_xyxy(boxes_xyxy, scores, iou_thr=nms_iou)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]

    # Sort by score desc & take top-K
    order = scores.argsort(descending=True)
    order = order[:max_bounding_boxes]
    boxes_xyxy = boxes_xyxy[order]
    scores = scores[order]
    return boxes_xyxy, scores

def _boxes_to_np_xyxy(boxes):
    # boxes may be torch tensor or numpy
    if hasattr(boxes, "detach"):
        b = boxes.detach().cpu().numpy()
    else:
        b = np.asarray(boxes)
    return b.astype(int)

def _filter_masks_by_area_quantiles(masks_bxhxw, boxes_xyxy,
                                    low_q=0.20, high_q=0.80,
                                    min_ratio=0.15, max_ratio=0.95):
    """
    masks_bxhxw: (B, H, W) binary {0,1} per-box masks in full-image coords
    boxes_xyxy: (B, 4) int [x1,y1,x2,y2]
    Returns: indices to keep (list), ratios (list)
    """
    boxes = _boxes_to_np_xyxy(boxes_xyxy)
    B = masks_bxhxw.shape[0]
    ratios = []
    for i in range(B):
        x1, y1, x2, y2 = boxes[i]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = max(x1+1, x2); y2 = max(y1+1, y2)
        roi = masks_bxhxw[i, y1:y2, x1:x2]
        box_area = max(1, (y2 - y1) * (x2 - x1))
        mask_area = int(roi.sum())
        ratios.append(mask_area / float(box_area))

    if len(ratios) >= 3:
        lo_q, hi_q = np.quantile(ratios, [low_q, high_q])
    else:
        lo_q, hi_q = min_ratio, max_ratio  # with 1-2 boxes, rely on absolute bounds

    keep = [i for i, r in enumerate(ratios)
            if (min_ratio <= r <= max_ratio) and (lo_q <= r <= hi_q)]

    if not keep:
        # fallback: index closest to midpoint of allowed range
        target = 0.5 * (min_ratio + max_ratio)
        keep = [int(np.argmin([abs(r - target) for r in ratios]))]
    return keep, ratios

def create_tensor_output(image_np, masks, boxes_filt):
    """
    Compatible with the original node's return convention.
    masks: a list/array where each item is stacked as (K, H, W) and will be unioned with np.any(..., axis=0)
    """
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.detach().cpu().numpy().astype(int) if boxes_filt is not None and hasattr(boxes_filt, "detach") else (boxes_filt.astype(int) if boxes_filt is not None else None)
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        # Normalize mask to 2D (H, W)
        mask_bool = mask.astype(bool)
        if mask_bool.ndim == 2:
            union_mask = mask_bool
        elif mask_bool.ndim == 3:
            # (K, H, W) -> (H, W)
            union_mask = np.any(mask_bool, axis=0)
        elif mask_bool.ndim == 4:
            # (B, K, H, W) -> (H, W)
            union_mask = np.any(mask_bool, axis=(0, 1))
        else:
            raise ValueError(f"Unsupported mask ndim: {mask_bool.ndim}, expected 2/3/4")

        image_np_copy[~union_mask] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(Image.fromarray(image_np_copy))
        output_masks.append(output_mask)
        output_images.append(output_image)
    return (output_images, output_masks)

def split_image_mask(image):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if "A" in image.getbands():
        mask = np.array(image.getchannel("A")).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        # return a zero mask the size of the image if no alpha
        h, w = image.size[1], image.size[0]
        mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")[None,]
    return (image_rgb, mask)


# -----------------------------
# V2 Node
# -----------------------------

class GroundingDinoSAM2SegmentV2:
    """
    Conservative V2:
      - GroundingDINO: scores + NMS + top-K
      - SAM: one mask per box
      - Filter by mask_area/box_area (drop extremes), union survivors
      - Defensive init for blob-detection points
      - Preview: kept boxes green, discarded red
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ("SAM2_MODEL", {}),
                "grounding_dino_model": ("GROUNDING_DINO_MODEL", {}),
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {}),
                "threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01},
                ),
                "max_bounding_boxes": (
                    "INT",
                    {"default": 4, "min": 1, "max": 10, "step": 1},
                ),
                "enable_blob_detection": (["true", "false"], {"default": "false"}),
                "light_hue_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "light_hue_max": ("FLOAT", {"default": 360.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "light_sat_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "light_val_min": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dark_hue_min": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "dark_hue_max": ("FLOAT", {"default": 150.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "dark_val_max": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_blob_size": ("INT", {"default": 100, "min": 10, "max": 10000, "step": 10}),
                "num_positive_points": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "num_negative_points": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
                "erosion_kernel": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            }
        }

    CATEGORY = "segment_anything2 (V2)"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt, threshold, max_bounding_boxes,
             enable_blob_detection, light_hue_min, light_hue_max, light_sat_min, light_val_min,
             dark_hue_min, dark_hue_max, dark_val_max, min_blob_size, num_positive_points,
             num_negative_points, erosion_kernel):

        from .node import get_control_points
        # Constants for conservative area filtering (middle quantile + absolute bounds)
        LOW_Q = 0.20
        HIGH_Q = 0.80
        MIN_RATIO = 0.15
        MAX_RATIO = 0.95

        res_images = []
        res_masks = []
        previews = []
        temp_path = folder_paths.get_temp_directory()

        for item in image:
            item = Image.fromarray(
                np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)
            ).convert("RGBA")

            boxes, scores = groundingdino_predict_v2(
                grounding_dino_model, item, prompt, threshold, max_bounding_boxes
            )
            if boxes.shape[0] == 0:
                continue

            # Blob detection points (defensive init)
            point_coords = None
            point_labels = None
            positive_points = []
            negative_points = []
            if enable_blob_detection == "true":
                img_np = np.array(item)[:, :, :3]  # RGB
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                light_lower = np.array([light_hue_min / 2, light_sat_min * 255, light_val_min * 255])
                light_upper = np.array([light_hue_max / 2, 255, 255])
                light_mask = cv2.inRange(hsv, light_lower, light_upper)

                dark_lower = np.array([dark_hue_min / 2, 0, 0])
                dark_upper = np.array([dark_hue_max / 2, 255, dark_val_max * 255])
                dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)

                kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)
                light_mask = cv2.erode(light_mask, kernel, iterations=1)
                light_mask = cv2.dilate(light_mask, kernel, iterations=1)
                dark_mask = cv2.erode(dark_mask, kernel, iterations=1)
                dark_mask = cv2.dilate(dark_mask, kernel, iterations=1)

                if num_positive_points > 0:
                    positive_points = get_control_points(light_mask, min_blob_size, num_positive_points)
                if num_negative_points > 0:
                    negative_points = get_control_points(dark_mask, min_blob_size, num_negative_points)

                if len(positive_points) or len(negative_points):
                    all_points = positive_points + negative_points
                    point_coords = np.array(all_points)
                    point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))

            # Create preview image with boxes (kept vs. discarded later)
            debug_img = np.array(item)[:, :, :3]  # RGB copy
            debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)

            # Run SAM (one mask per box)
            predictor = SAM2ImagePredictor(sam_model)
            image_np = np.array(item)
            image_np_rgb = image_np[..., :3]
            predictor.set_image(image_np_rgb)

            # predictor expects numpy boxes (N,4); ensure float32
            boxes_np = boxes.detach().cpu().numpy().astype(np.float32)

            # If both boxes and points are provided, tile points per box to match batch dim B
            if point_coords is not None:
                pc = np.asarray(point_coords)
                pl = np.asarray(point_labels) if point_labels is not None else None
                if pc.ndim == 2:
                    pc = pc[None, ...]  # (1, N, 2)
                if pl is not None and pl.ndim == 1:
                    pl = pl[None, ...]  # (1, N)
                B = boxes_np.shape[0]
                if pc.shape[0] != B:
                    pc = np.repeat(pc, B, axis=0)
                    if pl is not None:
                        pl = np.repeat(pl, B, axis=0)
                point_coords_in = pc.astype(np.float32)
                point_labels_in = pl.astype(np.int64) if pl is not None else None
            else:
                point_coords_in = None
                point_labels_in = None

            masks, sam_scores, _ = predictor.predict(
                point_coords=point_coords_in,
                point_labels=point_labels_in,
                box=boxes_np,
                multimask_output=False
            )

            # Normalize mask shape to (B, H, W)
            if masks.ndim == 2:
                masks = masks[None, ...]
            elif masks.ndim == 4:
                # (B, 1, H, W) -> squeeze
                masks = masks[:, 0, :, :]
            masks_b = (masks > 0).astype(np.uint8)

            # Area-based filtering across boxes
            keep_idx, ratios = _filter_masks_by_area_quantiles(
                masks_bxhxw=masks_b, boxes_xyxy=boxes,
                low_q=LOW_Q, high_q=HIGH_Q,
                min_ratio=MIN_RATIO, max_ratio=MAX_RATIO
            )

            # Draw kept (green) and discarded (red) boxes for preview
            boxes_int = boxes.detach().cpu().numpy().astype(int)
            for i, b in enumerate(boxes_int):
                x1, y1, x2, y2 = b
                color = (0, 255, 0) if i in keep_idx else (0, 0, 255)
                cv2.rectangle(debug_img_bgr, (x1, y1), (x2, y2), color, 2)

            if point_coords is not None:
                for i, pt in enumerate(point_coords):
                    color = (0, 255, 0) if point_labels[i] == 1 else (0, 0, 255)
                    cv2.circle(debug_img_bgr, (int(pt[0]), int(pt[1])), 5, color, -1)

            debug_img_rgb = cv2.cvtColor(debug_img_bgr, cv2.COLOR_BGR2RGB)
            fn = f"{uuid.uuid4()}.png"
            full_path = os.path.join(folder_paths.get_temp_directory(), fn)
            Image.fromarray(debug_img_rgb).save(full_path)
            previews.append({"filename": fn, "subfolder": "", "type": "temp"})

            # Build survivors stack (1, K, H, W) and create Comfy outputs
            survivors = masks_b[keep_idx, :, :]
            survivors = survivors[None, ...]  # (1, K, H, W)
            images_t, masks_t = create_tensor_output(image_np, [survivors], boxes)
            res_images.extend(images_t)
            res_masks.extend(masks_t)

        if len(res_images) == 0:
            # Produce an empty mask with the same spatial size as input
            _, height, width, _ = image.shape
            empty_mask = torch.zeros(
                (1, height, width), dtype=torch.uint8, device="cpu"
            )
            return {"ui": {"images": []}, "result": (empty_mask, empty_mask)}
        return {"ui": {"images": previews}, "result": (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))}


# =============================
# V2Ex helpers (do not override V2)
# =============================

def _load_dino_image_at_scale_v2ex(image_pil, short_side=800):
    transform = T.Compose(
        [
            T.RandomResize([int(short_side)], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil.convert("RGB"), None)
    return image

def _expand_and_clip_boxes_v2ex(boxes_xyxy_t, pad_frac, W, H):
    if boxes_xyxy_t.numel() == 0:
        return boxes_xyxy_t
    wh = torch.stack([(boxes_xyxy_t[:,2]-boxes_xyxy_t[:,0]), (boxes_xyxy_t[:,3]-boxes_xyxy_t[:,1])], dim=1)
    pad = wh * pad_frac
    x1 = (boxes_xyxy_t[:,0] - pad[:,0]).clamp(0, W-1)
    y1 = (boxes_xyxy_t[:,1] - pad[:,1]).clamp(0, H-1)
    x2 = (boxes_xyxy_t[:,2] + pad[:,0]).clamp(0, W-1)
    y2 = (boxes_xyxy_t[:,3] + pad[:,1]).clamp(0, H-1)
    return torch.stack([x1,y1,x2,y2], dim=1)

def _parse_float_list_v2ex(s, default_list=None):
    if s is None:
        return default_list[:] if default_list is not None else []
    s = str(s).strip()
    if not s:
        return default_list[:] if default_list is not None else []
    out = []
    for tok in s.split("|"):
        tok = tok.strip()
        try:
            out.append(float(tok))
        except Exception:
            pass
    # dedupe while preserving order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def _parse_int_list_v2ex(s, default_list=None):
    if s is None:
        return default_list[:] if default_list is not None else []
    s = str(s).strip()
    if not s:
        return default_list[:] if default_list is not None else []
    out = []
    for tok in s.split("|"):
        tok = tok.strip()
        try:
            out.append(int(tok))
        except Exception:
            pass
    # dedupe
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def _collect_dino_candidates_multi_v2ex(dino_model, image_pil, base_prompt, prompts_extra,
                                        thresholds, scales, nms_iou, max_after_merge):
    """
    Run DINO across (prompt in {base, variants}) x (thresholds) x (scales).
    Merge all boxes with a global NMS, keep top-K by score.
    Returns (boxes_xyxy_t, scores_t)
    """
    W, H = image_pil.size
    all_boxes = []
    all_scores = []

    prompts = [base_prompt] + [p for p in prompts_extra if p.strip()]
    for p in prompts:
        for thr in thresholds:
            for s in scales:
                # build DINO input at this scale
                dino_image = _load_dino_image_at_scale_v2ex(image_pil, short_side=s)
                # forward
                boxes_n, scores = _get_grounding_output_with_scores(dino_model, dino_image, p, thr)
                if boxes_n.numel() == 0:
                    continue
                boxes_xyxy = _xywhn_to_xyxy_abs(boxes_n, W, H)  # map to original coords (consistent with V2)
                all_boxes.append(boxes_xyxy)
                all_scores.append(scores)

    if not all_boxes:
        return torch.empty((0,4), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)

    boxes_cat = torch.cat(all_boxes, dim=0)
    scores_cat = torch.cat(all_scores, dim=0)

    # Global NMS to dedupe across passes
    keep = _nms_xyxy(boxes_cat, scores_cat, iou_thr=nms_iou)
    boxes_cat = boxes_cat[keep]
    scores_cat = scores_cat[keep]

    # Keep top-K by score
    order = scores_cat.argsort(descending=True)
    order = order[:max_after_merge]
    return boxes_cat[order], scores_cat[order]

def _tile_boxes_with_paddings_v2ex(boxes_xyxy_t, paddings, W, H):
    if boxes_xyxy_t.numel() == 0:
        return boxes_xyxy_t, []
    out_boxes = []
    pad_index = []
    for pi, pf in enumerate(paddings):
        out_boxes.append(_expand_and_clip_boxes_v2ex(boxes_xyxy_t, pf, W, H))
        pad_index.extend([pi] * boxes_xyxy_t.shape[0])
    return torch.cat(out_boxes, dim=0), pad_index

def _flatten_sam_masks_v2ex(masks, boxes_xyxy_t, sam_multimask):
    """
    Normalize SAM outputs to (N, H, W) and expand boxes to match N.
    If multimask=True and SAM returned (B, M, H, W), we reshape to (B*M, H, W).
    """
    if masks.ndim == 2:
        masks_b = masks[None, ...]
        boxes_rep = boxes_xyxy_t
    elif masks.ndim == 3:
        masks_b = masks
        boxes_rep = boxes_xyxy_t
    elif masks.ndim == 4:
        # (B, M, H, W) -> (B*M, H, W)
        B, M, H, W = masks.shape
        masks_b = masks.reshape(B*M, H, W)
        boxes_rep = boxes_xyxy_t.repeat_interleave(M, dim=0)
    else:
        raise ValueError(f"SAM returned unexpected ndim={masks.ndim}")
    masks_b = (masks_b > 0).astype(np.uint8)
    return masks_b, boxes_rep

# =============================
# V2Ex Node
# =============================

class GroundingDinoSAM2SegmentV2Ex:
    """
    V2Ex: diversity-first variant
      - Multi-pass GroundingDINO across thresholds/scales/optional prompt variants
      - Optional box paddings to produce multiple SAM inputs per detection
      - Optional SAM multimask_output to get multiple proposals per box
      - Area/box filtering (quantiles + absolute bounds), union survivors
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ("SAM2_MODEL", {}),
                "grounding_dino_model": ("GROUNDING_DINO_MODEL", {}),
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "max_bounding_boxes": ("INT", {"default": 4, "min": 1, "max": 30, "step": 1}),

                # Diversity controls
                "dino_thresholds": ("STRING", {"default": "0.25|0.30"}),
                "dino_scales": ("STRING", {"default": "736|800|896"}),
                "prompt_variants": ("STRING", {"default": ""}),  # e.g., "dalmatian dog|purple hair|tail"
                "box_paddings": ("STRING", {"default": "0.0|0.08"}),  # fraction of box size

                "sam_multimask_output": (["false", "true"], {"default": "true"}),
                "max_candidates_after_merge": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "max_sam_boxes": ("INT", {"default": 12, "min": 1, "max": 128, "step": 1}),

                # Area filter knobs
                "area_low_q": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "area_high_q": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01}),
                "area_min_ratio": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "area_max_ratio": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Blob detection (kept compatible)
                "enable_blob_detection": (["true", "false"], {"default": "false"}),
                "light_hue_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "light_hue_max": ("FLOAT", {"default": 360.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "light_sat_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "light_val_min": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dark_hue_min": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "dark_hue_max": ("FLOAT", {"default": 150.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "dark_val_max": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_blob_size": ("INT", {"default": 100, "min": 10, "max": 10000, "step": 10}),
                "num_positive_points": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "num_negative_points": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
                "erosion_kernel": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            }
        }

    CATEGORY = "segment_anything2 (V2+)"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt, threshold, max_bounding_boxes,
             dino_thresholds, dino_scales, prompt_variants, box_paddings,
             sam_multimask_output, max_candidates_after_merge, max_sam_boxes,
             area_low_q, area_high_q, area_min_ratio, area_max_ratio,
             enable_blob_detection, light_hue_min, light_hue_max, light_sat_min, light_val_min,
             dark_hue_min, dark_hue_max, dark_val_max, min_blob_size, num_positive_points,
             num_negative_points, erosion_kernel):

        from .node import get_control_points

        res_images = []
        res_masks = []
        previews = []
        temp_path = folder_paths.get_temp_directory()

        # Parse sweep inputs
        thresholds = _parse_float_list_v2ex(dino_thresholds, default_list=[threshold])
        if threshold not in thresholds:
            thresholds.append(float(threshold))
        thresholds = [float(np.clip(t, 0.0, 1.0)) for t in thresholds]

        scales = _parse_int_list_v2ex(dino_scales, default_list=[800])
        if 800 not in scales:
            scales.append(800)
        scales = [int(max(16, s)) for s in scales]

        paddings = _parse_float_list_v2ex(box_paddings, default_list=[0.0, 0.08])
        paddings = [float(max(0.0, p)) for p in paddings]

        prompts_extra = [p for p in (prompt_variants.split("|") if prompt_variants else [])]

        for item in image:
            item = Image.fromarray(
                np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)
            ).convert("RGBA")

            # 1) Collect diverse DINO candidates
            boxes_merged, scores_merged = _collect_dino_candidates_multi_v2ex(
                grounding_dino_model, item, prompt, prompts_extra, thresholds, scales,
                nms_iou=0.5, max_after_merge=max_candidates_after_merge
            )

            if boxes_merged.shape[0] == 0:
                continue

            # 2) Expand boxes with paddings to create more SAM inputs
            W, H = item.size
            boxes_padded, pad_index = _tile_boxes_with_paddings_v2ex(boxes_merged, paddings, W, H)

            # Cap number of boxes fed to SAM
            if boxes_padded.shape[0] > max_sam_boxes:
                # Select by interleaving across paddings to keep diversity
                idx = []
                n = min(max_sam_boxes, boxes_padded.shape[0])
                stride = max(1, boxes_padded.shape[0] // n)
                for i in range(0, boxes_padded.shape[0], stride):
                    idx.append(i)
                    if len(idx) >= n:
                        break
                boxes_padded = boxes_padded[idx]

            # 3) Blob detection points (defensive init)
            point_coords = None
            point_labels = None
            positive_points = []
            negative_points = []
            if enable_blob_detection == "true":
                img_np = np.array(item)[:, :, :3]  # RGB
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                light_lower = np.array([light_hue_min / 2, light_sat_min * 255, light_val_min * 255])
                light_upper = np.array([light_hue_max / 2, 255, 255])
                light_mask = cv2.inRange(hsv, light_lower, light_upper)

                dark_lower = np.array([dark_hue_min / 2, 0, 0])
                dark_upper = np.array([dark_hue_max / 2, 255, dark_val_max * 255])
                dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)

                kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)
                light_mask = cv2.erode(light_mask, kernel, iterations=1)
                light_mask = cv2.dilate(light_mask, kernel, iterations=1)
                dark_mask = cv2.erode(dark_mask, kernel, iterations=1)
                dark_mask = cv2.dilate(dark_mask, kernel, iterations=1)

                if num_positive_points > 0:
                    positive_points = get_control_points(light_mask, min_blob_size, num_positive_points)
                if num_negative_points > 0:
                    negative_points = get_control_points(dark_mask, min_blob_size, num_negative_points)

                if len(positive_points) or len(negative_points):
                    all_points = positive_points + negative_points
                    point_coords = np.array(all_points)
                    point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))

            # 4) Preview image (we'll color later after filtering)
            debug_img = np.array(item)[:, :, :3]  # RGB copy
            debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            boxes_int_all = boxes_padded.detach().cpu().numpy().astype(int)
            for b in boxes_int_all:
                x1,y1,x2,y2 = b
                cv2.rectangle(debug_img_bgr, (x1,y1), (x2,y2), (255, 165, 0), 1)  # orange for "candidates"

            # 5) Run SAM
            predictor = SAM2ImagePredictor(sam_model)
            image_np = np.array(item)
            image_np_rgb = image_np[..., :3]
            predictor.set_image(image_np_rgb)

            boxes_np = boxes_padded.detach().cpu().numpy().astype(np.float32)
            multimask_flag = (sam_multimask_output == "true")
            # If both boxes and points are provided, tile points per box to match batch dim B
            if point_coords is not None:
                pc = np.asarray(point_coords)
                pl = np.asarray(point_labels) if point_labels is not None else None
                if pc.ndim == 2:
                    pc = pc[None, ...]  # (1, N, 2)
                if pl is not None and pl.ndim == 1:
                    pl = pl[None, ...]  # (1, N)
                B = boxes_np.shape[0]
                if pc.shape[0] != B:
                    pc = np.repeat(pc, B, axis=0)
                    if pl is not None:
                        pl = np.repeat(pl, B, axis=0)
                point_coords_in = pc.astype(np.float32)
                point_labels_in = pl.astype(np.int64) if pl is not None else None
            else:
                point_coords_in = None
                point_labels_in = None

            masks, sam_scores, _ = predictor.predict(
                point_coords=point_coords_in,
                point_labels=point_labels_in,
                box=boxes_np,
                multimask_output=multimask_flag
            )

            masks_b, boxes_rep = _flatten_sam_masks_v2ex(masks, boxes_padded, multimask_flag)

            # 6) Area-based filtering across proposals
            keep_idx, ratios = _filter_masks_by_area_quantiles(
                masks_bxhxw=masks_b, boxes_xyxy=boxes_rep,
                low_q=area_low_q, high_q=area_high_q,
                min_ratio=area_min_ratio, max_ratio=area_max_ratio
            )

            # Draw kept (green) and discarded (red) boxes for preview
            keep_set = set(keep_idx)
            # Since boxes_rep may be larger than boxes_padded (if multimask=True), we only draw per unique box
            # Use the first occurrence of each box index
            boxes_unique = boxes_padded.detach().cpu().numpy().astype(int)
            used = set()
            for i, b in enumerate(boxes_unique):
                x1,y1,x2,y2 = b
                if tuple(b) in used:
                    continue
                used.add(tuple(b))
                # If any proposal for this box survived, mark as green; else red
                # Map from proposals back to original box via integer division of i by M isn't trivial here;
                # As an approximation: if any proposal index with same box coords survived -> green
                # We'll check all proposals and compare coords
                survived = False
                for j in keep_idx:
                    # j indexes into masks_b and boxes_rep
                    bb = boxes_rep[j].detach().cpu().numpy().astype(int)
                    if (bb == b).all():
                        survived = True
                        break
                color = (0,255,0) if survived else (0,0,255)
                cv2.rectangle(debug_img_bgr, (x1,y1), (x2,y2), color, 2)

            if point_coords is not None:
                for i, pt in enumerate(point_coords):
                    color = (0, 255, 0) if point_labels[i] == 1 else (0, 0, 255)
                    cv2.circle(debug_img_bgr, (int(pt[0]), int(pt[1])), 5, color, -1)

            debug_img_rgb = cv2.cvtColor(debug_img_bgr, cv2.COLOR_BGR2RGB)
            fn = f"{uuid.uuid4()}.png"
            full_path = os.path.join(temp_path, fn)
            Image.fromarray(debug_img_rgb).save(full_path)
            previews.append({"filename": fn, "subfolder": "", "type": "temp"})

            # 7) Build survivors stack (1, K, H, W) and create Comfy outputs
            survivors = masks_b[keep_idx, :, :]
            if survivors.ndim == 2:
                survivors = survivors[None, ...]  # (1, H, W)
            else:
                survivors = survivors[None, ...]  # (1, K, H, W)
            images_t, masks_t = create_tensor_output(image_np, [survivors], boxes_rep)
            res_images.extend(images_t)
            res_masks.extend(masks_t)

        if len(res_images) == 0:
            _, height, width, _ = image.shape
            empty_mask = torch.zeros(
                (1, height, width), dtype=torch.uint8, device="cpu"
            )
            return {"ui": {"images": []}, "result": (empty_mask, empty_mask)}
        return {"ui": {"images": previews}, "result": (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))}
