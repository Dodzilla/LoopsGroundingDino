
import os
import uuid
import copy
import numpy as np
import cv2
import torch
from PIL import Image

import folder_paths
import comfy.model_management

# External modules already present in the original node package
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
        # union across K (axis=0)
        union_mask = np.any(mask, axis=0)
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

def get_centroids(mask, min_size, max_num):
    valid_centroids = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_idx = np.argsort(-areas)
        for idx in sorted_idx:
            if areas[idx] >= min_size:
                cx, cy = centroids[idx + 1]
                valid_centroids.append([cx, cy])
            if len(valid_centroids) >= max_num:
                break
    return valid_centroids

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
                    positive_points = get_centroids(light_mask, min_blob_size, num_positive_points)
                if num_negative_points > 0:
                    negative_points = get_centroids(dark_mask, min_blob_size, num_negative_points)

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
            masks, sam_scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
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

