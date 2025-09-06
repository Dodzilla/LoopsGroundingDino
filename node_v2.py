import os
import sys
import uuid
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import copy
import torch
import numpy as np
from PIL import Image
import logging
from torch.hub import download_url_to_file
import math
from urllib.parse import urlparse
import folder_paths
import comfy.model_management
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import (
    clean_state_dict as local_groundingdino_clean_state_dict,
)
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model
import glob
import folder_paths
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

logger = logging.getLogger("ComfyUI-SAM2")

sam_model_dir_name = "sam2"
sam_model_list = {
    "sam2_hiera_tiny": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    },
    "sam2_hiera_small.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    },
    "sam2_hiera_base_plus.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
    },
    "sam2_hiera_large.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    },
    "sam2_1_hiera_tiny.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    },
    "sam2_1_hiera_small.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    },
    "sam2_1_hiera_base_plus.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    },
    "sam2_1_hiera_large.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    },
}

groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
    },
}


def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, "bert-base-uncased")
    if glob.glob(
        os.path.join(comfy_bert_model_base, "**/model.safetensors"), recursive=True
    ):
        print("grounding-dino is using models/bert-base-uncased")
        return comfy_bert_model_base
    return "bert-base-uncased"


def list_files(dirpath, extensions=[]):
    return [
        f
        for f in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, f)) and f.split(".")[-1] in extensions
    ]


def list_sam_model():
    return list(sam_model_list.keys())


def load_sam_model(model_name):
    sam2_checkpoint_path = get_local_filepath(
        sam_model_list[model_name]["model_url"], sam_model_dir_name
    )
    model_file_name = os.path.basename(sam2_checkpoint_path)
    model_file_name = model_file_name.replace("2.1", "2_1")
    model_type = model_file_name.split(".")[0]

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    config_path = "sam2_configs"
    initialize(config_path=config_path)
    model_cfg = f"{model_type}.yaml"

    sam_device = comfy.model_management.get_torch_device()
    sam = build_sam2(model_cfg, sam2_checkpoint_path, device=sam_device)
    sam.model_name = model_file_name
    return sam


def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f"using extra model: {destination}")
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f"downloading {url} to {destination}")
        download_url_to_file(url, destination)
    return destination


def load_groundingdino_model(model_name):
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            groundingdino_model_dir_name,
        ),
    )

    if dino_model_args.text_encoder_type == "bert-base-uncased":
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()

    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            groundingdino_model_dir_name,
        ),
    )
    dino.load_state_dict(
        local_groundingdino_clean_state_dict(checkpoint["model"]), strict=False
    )
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    return dino


def list_groundingdino_model():
    return list(groundingdino_model_list.keys())


def groundingdino_predict(dino_model, image, prompt, threshold):
    def load_dino_image(image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        return boxes_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(dino_model, dino_image, prompt, threshold)
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt


def create_pil_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        output_masks.append(Image.fromarray(np.any(mask, axis=0)))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_images.append(Image.fromarray(image_np_copy))
    return output_images, output_masks


def create_tensor_output(image_np_rgba, masks, boxes_filt):
    """
    Expects RGBA input (we write 4 zeros to outside pixels).
    """
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np_rgba)  # keep 4 channels
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0], dtype=image_np_copy.dtype)
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
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)


# ---------- Control point utilities ----------

def get_control_points(mask, min_size, max_num, min_sep=12, min_dt=0.5, enforce_coverage=True):
    """
    Robust interior-point picker (component-aware).
      * If enforce_coverage=True  : guarantee ≥1 point per large component (may expand budget).
      * If enforce_coverage=False : obey max_num strictly (used for NEGATIVES).
    Returns: list[[x, y], ...] in image coords
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    comp_candidates = {}   # comp_id -> [(v,x,y), ...] sorted by distance desc
    small_candidates = []  # [(v,x,y,comp_id), ...] for very small comps (thin tips)

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        comp = (labels == lbl).astype(np.uint8)
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)

        if area < min_size:
            # safe single-point fallback for a few tiny regions
            yx = np.unravel_index(np.argmax(dt), dt.shape)
            y_s, x_s = int(yx[0]), int(yx[1])
            small_candidates.append((float(dt[y_s, x_s]), x_s, y_s, lbl))
            continue

        k = max(3, 2 * int(min_sep) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        dt_dil = cv2.dilate(dt, kernel)
        maxima = (dt >= (dt_dil - 1e-6)) & (dt >= float(min_dt))
        ys, xs = np.where(maxima)
        vals = dt[ys, xs]
        pts = [(float(v), int(x), int(y)) for x, y, v in zip(xs, ys, vals)]
        if len(pts) == 0:
            # ultra-thin shape: still take its farthest pixel
            yx = np.unravel_index(np.argmax(dt), dt.shape)
            y_f, x_f = int(yx[0]), int(yx[1])
            pts = [(float(dt[y_f, x_f]), x_f, y_f)]
        pts.sort(key=lambda t: -t[0])
        comp_candidates[lbl] = pts

    num_comps = len(comp_candidates)
    num_small = len(small_candidates)

    # Budget policy
    target_max = max_num
    if enforce_coverage:
        cover_small = min(num_small, 3)  # guard against noisy specks
        need = num_comps + cover_small
        if need > target_max:
            target_max = need

    chosen = []
    chosen_labels = set()

    # First pass: 1 per large component (best interior)
    for comp_id, cand_list in comp_candidates.items():
        if len(chosen) >= target_max:
            break
        v, x, y = cand_list[0]
        chosen.append((x, y, comp_id))
        chosen_labels.add(comp_id)

    # Optional: include a few tiny components (≤3)
    if enforce_coverage and num_small > 0 and num_small <= 3:
        for _, x, y, comp_id in small_candidates:
            if len(chosen) >= target_max:
                break
            chosen.append((int(x), int(y), comp_id))
            chosen_labels.add(comp_id)

    # Second pass: fill leftover budget via global NMS
    remaining = []
    for comp_id, cand_list in comp_candidates.items():
        for idx, (v, x, y) in enumerate(cand_list):
            if idx == 0 and comp_id in chosen_labels:
                continue
            remaining.append((float(v), int(x), int(y), comp_id))
    remaining.sort(key=lambda t: -t[0])

    min_sep2 = float(min_sep) * float(min_sep)
    for v, x, y, comp_id in remaining:
        if len(chosen) >= target_max:
            break
        ok = True
        for cx, cy, _ in chosen:
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy < min_sep2:
                ok = False
                break
        if ok:
            chosen.append((x, y, comp_id))

    return [[float(x), float(y)] for (x, y, _) in chosen]


# ---- Scored negative point selection helpers ----

def _angular_dist_to_band_deg(H_deg, band_min_deg, band_max_deg):
    """
    Minimal circular (0..360) distance in degrees from each hue H to the closed band [min,max].
    Vectorized for numpy arrays.
    """
    # Ensure arrays
    H = np.asarray(H_deg, dtype=np.float32)
    a = float(band_min_deg)
    b = float(band_max_deg)
    # inside-band mask
    inside = (H >= a) & (H <= b)
    # distance to nearest edge (wrap-around)
    d_low  = (a - H) % 360.0
    d_high = (H - b) % 360.0
    dist = np.minimum(d_low, d_high)
    dist[inside] = 0.0
    return dist


def _nms_pick_topk(score_map_f32, k, min_sep_px):
    """
    Greedy NMS picker on a float32 score map.
    Returns list of (x, y) coordinates (float).
    """
    pts = []
    if k <= 0:
        return pts
    s = score_map_f32.copy()
    s[s < 0] = 0.0
    if s.max() <= 0:
        return pts
    r = max(1, int(min_sep_px))
    for _ in range(k):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(s)
        if max_val <= 1e-8:
            break
        x, y = int(max_loc[0]), int(max_loc[1])
        pts.append((float(x), float(y)))
        # Suppress a disk around the chosen point
        cv2.circle(s, (x, y), r, 0.0, -1)
    return pts


def pick_negative_points_scored(hsv_roi, lm_roi, dm_roi, k, band_min_deg, band_max_deg, min_sep_px, safe_erode=1):
    """
    Choose up to k negative points using a score that prefers:
      - hues far from the croc band,
      - pixels farther from the croc-color mask.
    """
    if k <= 0:
        return []

    # HSV channels (OpenCV H in [0..180]); convert H to degrees [0..360)
    H = hsv_roi[..., 0].astype(np.float32) * 2.0
    # Candidate region: inverse hue minus positive mask
    cand = (dm_roi > 0).astype(np.uint8)
    # Soft safety buffer around positive mask
    if safe_erode > 0:
        lm_dil = cv2.dilate((lm_roi > 0).astype(np.uint8), np.ones((safe_erode, safe_erode), np.uint8), iterations=1)
        cand = cv2.bitwise_and(cand, cv2.bitwise_not(lm_dil))

    # Distance-from-band (bigger = farther from croc hues)
    hue_dist = _angular_dist_to_band_deg(H, band_min_deg, band_max_deg)  # degrees
    hue_dist_norm = (hue_dist / 180.0).astype(np.float32)

    # Distance-from-croc-mask (within ROI)
    inv_lm = (lm_roi == 0).astype(np.uint8)
    dt = cv2.distanceTransform(inv_lm, cv2.DIST_L2, 5).astype(np.float32)
    if dt.max() > 0:
        dt /= dt.max()
    # Compose score (weights: hue dominates)
    score = 0.75 * hue_dist_norm + 0.25 * dt
    # Zero out non-candidates
    score[cand == 0] = 0.0

    # Pick top-k with NMS
    pts = _nms_pick_topk(score, k, min_sep_px)
    return pts


def _ensure_3d_mask(m):
    # normalize SAM outputs to (K, H, W) float/bool
    if m.ndim == 2:
        return m[None, :, :]
    elif m.ndim == 3:
        return m
    elif m.ndim == 4:
        return m[0, ...]
    else:
        raise ValueError(f"Unexpected mask ndim={m.ndim}")


class SAM2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_sam_model(),),
            }
        }

    CATEGORY = "segment_anything2"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM2_MODEL",)

    def main(self, model_name):
        sam_model = load_sam_model(model_name)
        return (sam_model,)


class GroundingDinoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_groundingdino_model(),),
            }
        }

    CATEGORY = "segment_anything2"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL",)

    def main(self, model_name):
        dino_model = load_groundingdino_model(model_name)
        return (dino_model,)


class GroundingDinoSAM2SegmentV2:
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
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
                "enable_blob_detection": (["true", "false"], {"default": "false"}),
                # Positive (croc) band — covers greens + yellow/olive belly
                "light_hue_min": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "light_hue_max": ("FLOAT", {"default": 150.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "light_sat_min": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "light_val_min": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                # If dark_hue_max - dark_hue_min < 2°, we auto-use inverse-of-positive with a small gap
                "dark_hue_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "dark_hue_max": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 360.0, "step": 1.0}),
                "dark_val_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_blob_size": ("INT", {"default": 1800, "min": 10, "max": 10000, "step": 10}),
                "num_positive_points": ("INT", {"default": 8, "min": 0, "max": 20, "step": 1}),
                "num_negative_points": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "erosion_kernel": ("INT", {"default": 3, "min": 1, "max": 15, "step": 1}),
            }
        }

    CATEGORY = "segment_anything2"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt, threshold, max_bounding_boxes,
             enable_blob_detection, light_hue_min, light_hue_max, light_sat_min, light_val_min,
             dark_hue_min, dark_hue_max, dark_val_max, min_blob_size, num_positive_points,
             num_negative_points, erosion_kernel):

        res_images = []
        res_masks = []
        previews = []
        temp_path = folder_paths.get_temp_directory()

        # Inverse-mode gap (deg) when we auto-compute negatives as inverse hues
        inverse_gap_deg = 12.0

        for item in image:
            # tensor -> PIL RGBA
            item = Image.fromarray(
                np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)
            ).convert("RGBA")

            # GroundingDINO proposals
            boxes = groundingdino_predict(grounding_dino_model, item, prompt, threshold)
            if boxes.shape[0] == 0:
                continue
            if boxes.shape[0] > max_bounding_boxes:
                boxes = boxes[:max_bounding_boxes]

            # Prepare buffers
            image_np_rgba = np.array(item)          # (H,W,4)
            image_np_rgb  = image_np_rgba[..., :3]  # (H,W,3)
            H_img, W_img = image_np_rgb.shape[:2]

            # Resolution-aware scaling (keeps behavior consistent across sizes)
            scale_factor = math.sqrt((H_img * W_img) / float(512 * 512))
            k = max(1, int(round(erosion_kernel * scale_factor)))
            kernel = np.ones((k, k), np.uint8)
            area_ratio = (H_img * W_img) / float(512 * 512)
            scaled_min_blob = max(10, int(min_blob_size * area_ratio))

            # HSV once (global), crop per box
            hsv = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2HSV)

            # Positive croc mask
            light_lower = np.array([light_hue_min / 2.0, light_sat_min * 255.0, light_val_min * 255.0])
            light_upper = np.array([light_hue_max / 2.0, 255.0, 255.0])
            light_mask_full = cv2.inRange(hsv, light_lower, light_upper)
            light_mask_full = cv2.erode(light_mask_full, kernel, iterations=1)
            light_mask_full = cv2.dilate(light_mask_full, kernel, iterations=1)

            # Negative mask (inverse or explicit)
            use_inverse = (abs(dark_hue_max - dark_hue_min) < 2.0)
            if use_inverse:
                gap_min = max(0.0, light_hue_min - inverse_gap_deg)
                gap_max = min(360.0, light_hue_max + inverse_gap_deg)
                mask_neg1 = None
                mask_neg2 = None
                v_max = 255.0 * dark_val_max
                if gap_min > 0.0:
                    lower1 = np.array([0.0, 0.0, 0.0])
                    upper1 = np.array([(gap_min - 1e-6) / 2.0, 255.0, v_max])
                    mask_neg1 = cv2.inRange(hsv, lower1, upper1)
                if gap_max < 360.0:
                    lower2 = np.array([(gap_max + 1e-6) / 2.0, 0.0, 0.0])
                    upper2 = np.array([180.0, 255.0, v_max])
                    mask_neg2 = cv2.inRange(hsv, lower2, upper2)
                if mask_neg1 is not None and mask_neg2 is not None:
                    dark_mask_full = cv2.bitwise_or(mask_neg1, mask_neg2)
                elif mask_neg1 is not None:
                    dark_mask_full = mask_neg1
                elif mask_neg2 is not None:
                    dark_mask_full = mask_neg2
                else:
                    dark_mask_full = np.zeros_like(light_mask_full)
            else:
                dark_lower = np.array([dark_hue_min / 2.0, 0.0, 0.0])
                dark_upper = np.array([dark_hue_max / 2.0, 255.0, dark_val_max * 255.0])
                dark_mask_full = cv2.inRange(hsv, dark_lower, dark_upper)

            dark_mask_full = cv2.erode(dark_mask_full, kernel, iterations=1)
            dark_mask_full = cv2.dilate(dark_mask_full, kernel, iterations=1)

            # Debug preview
            debug_img_bgr = cv2.cvtColor(image_np_rgb.copy(), cv2.COLOR_RGB2BGR)
            boxes_np = boxes.numpy().astype(int)
            for x1, y1, x2, y2 in boxes_np:
                cv2.rectangle(debug_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # SAM2 predictor
            predictor = SAM2ImagePredictor(sam_model)
            predictor.set_image(image_np_rgb)

            per_box_masks = []
            for (x1, y1, x2, y2) in boxes_np:
                # clamp
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W_img, x2); y2 = min(H_img, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                pc_i = pl_i = None
                if enable_blob_detection == "true":
                    # Crop ROI masks to the box
                    lm = light_mask_full[y1:y2, x1:x2]
                    dm = dark_mask_full[y1:y2, x1:x2]
                    hsv_roi = hsv[y1:y2, x1:x2, :]

                    pos_pts_abs = []
                    neg_pts_abs = []

                    # POSITIVES: robust interior points with island coverage
                    if num_positive_points > 0:
                        pos_pts_roi = get_control_points(
                            lm,
                            scaled_min_blob,
                            num_positive_points,
                            min_sep=max(12, int(round(3 * scale_factor * erosion_kernel))),
                            enforce_coverage=True
                        )
                        pos_pts_abs = [[x1 + float(px), y1 + float(py)] for (px, py) in pos_pts_roi]

                    # NEGATIVES: scored selection, strictly capped
                    if num_negative_points > 0:
                        # safety_erode scales a small buffer so we don't sample right on croc borders
                        safe_erode = max(1, int(round(1.5 * scale_factor)))
                        neg_pts_roi = pick_negative_points_scored(
                            hsv_roi, lm, dm, num_negative_points,
                            light_hue_min, light_hue_max,
                            min_sep_px=max(8, int(round(2 * scale_factor * erosion_kernel))),
                            safe_erode=safe_erode
                        )

                        # Fallback if scored method found nothing: try ring∩dark for robustness
                        if len(neg_pts_roi) < num_negative_points:
                            dil = cv2.dilate(lm, kernel, iterations=1)
                            ring = cv2.subtract(dil, lm)
                            ring_and_dark = cv2.bitwise_and(ring, dm)
                            remaining = num_negative_points - len(neg_pts_roi)
                            more = get_control_points(
                                ring_and_dark, 3, remaining,
                                min_sep=max(8, int(round(2 * scale_factor * erosion_kernel))),
                                enforce_coverage=False
                            )
                            neg_pts_roi += more

                        neg_pts_abs = [[x1 + float(nx), y1 + float(ny)] for (nx, ny) in neg_pts_roi]

                    all_pts = pos_pts_abs + neg_pts_abs
                    if all_pts:
                        pc_i = np.array(all_pts, dtype=np.float32)
                        pl_i = np.array([1] * len(pos_pts_abs) + [0] * len(neg_pts_abs), dtype=np.int64)

                        # draw points on debug
                        for (px, py), lbl in zip(pc_i, pl_i):
                            color = (0, 255, 0) if int(lbl) == 1 else (0, 0, 255)
                            cv2.circle(debug_img_bgr, (int(px), int(py)), 5, color, -1)

                masks_i, scores, _ = predictor.predict(
                    point_coords=pc_i, point_labels=pl_i,
                    box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                    multimask_output=False
                )
                masks_i = _ensure_3d_mask(masks_i)
                per_box_masks.append(masks_i)

            # Save preview
            debug_rgb = cv2.cvtColor(debug_img_bgr, cv2.COLOR_BGR2RGB)
            fn = f"{uuid.uuid4()}.png"
            Image.fromarray(debug_rgb).save(os.path.join(temp_path, fn))
            previews.append({"filename": fn, "subfolder": "", "type": "temp"})

            # Build outputs (IMPORTANT: pass RGBA here)
            if len(per_box_masks) > 0:
                images, masks = create_tensor_output(image_np_rgba, per_box_masks, boxes)
                res_images.extend(images)
                res_masks.extend(masks)

        if len(res_images) == 0:
            _, height, width, _ = image.shape
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return {"ui": {"images": []}, "result": (empty_mask, empty_mask)}
        return {"ui": {"images": previews}, "result": (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))}