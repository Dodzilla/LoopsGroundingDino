
import os
import sys
import uuid
import cv2
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import copy
import torch
import numpy as np
from PIL import Image
import logging
from torch.hub import download_url_to_file
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


def create_tensor_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
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


def get_control_points(mask, min_size, max_num, min_sep=12, min_dt=0.5):
    """
    Robust interior-point picker (enhanced for wide distribution):
      - Works per connected component (8-connectivity)
      - Uses distance transform local maxima inside each component
      - Greedy NMS with component-aware logic enforces min separation (min_sep px)
    Returns: list[[x, y], ...] in image coordinates
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    comp_candidates = {}    # map comp_id -> list of (dt_value, x, y) for that component
    small_candidates = []   # list of (dt_value, x, y, comp_id) for very small components (area < min_size)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_size:
            # **UPGRADE 4:** Safe fallback for thin/isolated small regions – collect a point if few small comps
            comp_mask = (labels == lbl).astype(np.uint8)
            dt_small = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            # Take the farthest in-mask pixel (max distance) as representative point
            yx = np.unravel_index(np.argmax(dt_small), dt_small.shape)
            y_s, x_s = int(yx[0]), int(yx[1])
            small_candidates.append((float(dt_small[y_s, x_s]), x_s, y_s, lbl))
            continue

        # Compute distance transform for this component (for interior point detection)
        comp_mask = (labels == lbl).astype(np.uint8)
        dt = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        # Determine local maxima in the distance map (within this component)
        k = max(3, 2 * int(min_sep) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        dt_dil = cv2.dilate(dt, kernel)
        maxima_mask = (dt >= dt_dil - 1e-6) & (dt >= float(min_dt))
        ys, xs = np.where(maxima_mask)
        vals = dt[ys, xs]
        # Collect all candidate points for this component
        comp_points = []
        for x, y, v in zip(xs, ys, vals):
            comp_points.append((float(v), int(x), int(y)))
        if len(comp_points) == 0:
            # Fallback: no local max found (extremely thin shape) – use global farthest point
            yx = np.unravel_index(np.argmax(dt), dt.shape)
            y_f, x_f = int(yx[0]), int(yx[1])
            comp_points.append((float(dt[y_f, x_f]), x_f, y_f))
        comp_points.sort(key=lambda t: -t[0])  # sort candidates by distance (desc)
        comp_candidates[lbl] = comp_points

    # **UPGRADE 1:** Ensure each significant component yields at least one positive point
    # If more components than max_num, extend max_num to cover all components (wide spatial coverage)
    num_comps = len(comp_candidates)
    num_small = len(small_candidates)
    if num_comps + (num_small if num_small <= 3 else 0) > max_num:
        max_num = num_comps + (num_small if num_small <= 3 else 0)

    chosen = []
    chosen_labels = set()
    # Select one deepest point from each large component (guarantee one per comp)
    for comp_id, cand_list in comp_candidates.items():
        if len(chosen) >= max_num:
            break
        best_val, best_x, best_y = cand_list[0]  # highest-distance point in this comp
        chosen.append((best_x, best_y, comp_id))
        chosen_labels.add(comp_id)
    # Include fallback points for small components if the count is safe (<=3) to avoid false positives
    if num_small > 0 and num_small <= 3:
        for val, x, y, comp_id in small_candidates:
            if len(chosen) >= max_num:
                break
            chosen.append((int(x), int(y), comp_id))
            chosen_labels.add(comp_id)

    # **UPGRADE 1 (continued):** Additional points selection with global NMS for wide coverage
    # Prepare remaining candidates (excluding already chosen ones) across all components
    remaining_cands = []
    for comp_id, cand_list in comp_candidates.items():
        for val, x, y in cand_list:
            # Skip any point that was already chosen as the primary for its component
            if comp_id in chosen_labels and (x, y, comp_id) in chosen:
                continue
            remaining_cands.append((float(val), int(x), int(y), comp_id))
    # (Small components have no additional candidates beyond their one chosen point)

    remaining_cands.sort(key=lambda t: -t[0])  # sort globally by distance (desc)
    min_sep_sq = float(min_sep) * float(min_sep)
    for val, x, y, comp_id in remaining_cands:
        if len(chosen) >= max_num:
            break
        ok = True
        for cx, cy, cid in chosen:
            # Enforce minimum separation for additional points (across all comps)
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy < min_sep_sq:
                ok = False
                break
        if ok:
            chosen.append((x, y, comp_id))
            chosen_labels.add(comp_id)
    # Return point coordinates (float) without labels
    return [[float(x), float(y)] for (x, y, cid) in chosen]



def _ensure_3d_mask(m):
    # normalize SAM outputs to (K, H, W) float/bool
    if m.ndim == 2:
        return m[None, :, :]
    elif m.ndim == 3:
        # either (K, H, W) or (1, H, W) already
        return m
    elif m.ndim == 4:
        # (B, K, H, W) -> strip batch dim
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
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_bounding_boxes": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "enable_blob_detection": (["true", "false"], {"default": "false"}),
                "light_hue_min": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 360.0, "step": 1.0}),  # Crocodile primary hue range start (greenish)
                "light_hue_max": ("FLOAT", {"default": 125.0, "min": 0.0, "max": 360.0, "step": 1.0}),  # Crocodile primary hue range end (light green/yellow)
                "light_sat_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "light_val_min": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dark_hue_min": ("FLOAT", {"default": 135.0, "min": 0.0, "max": 360.0, "step": 1.0}),   # Inverted hue range start (after gap)
                "dark_hue_max": ("FLOAT", {"default": 360.0, "min": 0.0, "max": 360.0, "step": 1.0}),   # Inverted hue range end (covers rest of spectrum)
                "dark_val_max": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hue_gap": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 180.0, "step": 1.0}),         # **UPGRADE 2:** Minimum hue gap (degrees) between croc and non-croc masks
                "min_blob_size": ("INT", {"default": 100, "min": 10, "max": 10000, "step": 10}),
                "num_positive_points": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "num_negative_points": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
                "erosion_kernel": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            }
        }

    CATEGORY = "segment_anything2"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt, threshold, max_bounding_boxes,
            enable_blob_detection, light_hue_min, light_hue_max, light_sat_min, light_val_min,
            dark_hue_min, dark_hue_max, dark_val_max, hue_gap, min_blob_size,
            num_positive_points, num_negative_points, erosion_kernel):

        res_images, res_masks, previews = [], [], []
        temp_path = folder_paths.get_temp_directory()

        for item in image:
            # ---- tensor -> PIL ----
            item = Image.fromarray(
                np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)
            ).convert("RGBA")

            # ---- GroundingDINO (boxes) ----
            boxes = groundingdino_predict(grounding_dino_model, item, prompt, threshold)
            if boxes.shape[0] == 0:
                continue
            if boxes.shape[0] > max_bounding_boxes:
                boxes = boxes[:max_bounding_boxes]

            # ---- Prepare buffers (keep BOTH RGBA and RGB) ----
            image_np_rgba = np.array(item)           # 4ch (needed for create_tensor_output)
            image_np_rgb  = image_np_rgba[..., :3]   # 3ch (for HSV + SAM)
            H_img, W_img  = image_np_rgb.shape[:2]

            # ---- Resolution-aware morphology ----
            import math
            scale_factor = math.sqrt((H_img * W_img) / float(512 * 512))
            k = max(1, int(round(erosion_kernel * scale_factor)))
            kernel = np.ones((k, k), np.uint8)
            area_ratio = (H_img * W_img) / float(512 * 512)
            scaled_min_blob = max(10, int(min_blob_size * area_ratio))

            # ---- HSV masks (positive croc hues, negative = inverse hues with a gap) ----
            hsv = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2HSV)

            # positive (croc color band)
            light_lower = np.array([light_hue_min / 2.0, light_sat_min * 255.0, light_val_min * 255.0])
            light_upper = np.array([light_hue_max / 2.0, 255.0, 255.0])
            light_mask_full = cv2.inRange(hsv, light_lower, light_upper)
            light_mask_full = cv2.erode(light_mask_full, kernel, iterations=1)
            light_mask_full = cv2.dilate(light_mask_full, kernel, iterations=1)

            # negative = hues outside [light_hue_min - hue_gap, light_hue_max + hue_gap]
            gap_min = max(0.0, light_hue_min - hue_gap)
            gap_max = min(360.0, light_hue_max + hue_gap)

            mask_neg1 = None
            mask_neg2 = None
            if gap_min > 0:
                lower1 = np.array([0.0, 0.0, 0.0])
                upper1 = np.array([(gap_min - 1e-6) / 2.0, 255.0, 255.0])    # allow any V (not just dark)
                mask_neg1 = cv2.inRange(hsv, lower1, upper1)
            if gap_max < 360:
                lower2 = np.array([(gap_max + 1e-6) / 2.0, 0.0, 0.0])
                upper2 = np.array([180.0, 255.0, 255.0])
                mask_neg2 = cv2.inRange(hsv, lower2, upper2)
            if mask_neg1 is not None and mask_neg2 is not None:
                dark_mask_full = cv2.bitwise_or(mask_neg1, mask_neg2)
            elif mask_neg1 is not None:
                dark_mask_full = mask_neg1
            elif mask_neg2 is not None:
                dark_mask_full = mask_neg2
            else:
                dark_mask_full = np.zeros_like(light_mask_full)

            # small refine on negatives (keeps edges clean)
            dark_mask_full = cv2.erode(dark_mask_full, kernel, iterations=1)
            dark_mask_full = cv2.dilate(dark_mask_full, kernel, iterations=1)

            # ---- Debug preview canvas ----
            debug_img_bgr = cv2.cvtColor(image_np_rgb.copy(), cv2.COLOR_RGB2BGR)
            boxes_np = boxes.numpy().astype(int)
            for x1, y1, x2, y2 in boxes_np:
                cv2.rectangle(debug_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ---- SAM2 ----
            predictor = SAM2ImagePredictor(sam_model)
            predictor.set_image(image_np_rgb)

            per_box_masks = []
            for x1, y1, x2, y2 in boxes_np:
                # clamp
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W_img, x2); y2 = min(H_img, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                pc_i = pl_i = None
                if enable_blob_detection == "true":
                    roi_light = light_mask_full[y1:y2, x1:x2]
                    roi_dark  = dark_mask_full[y1:y2, x1:x2]

                    pos_pts_abs, neg_pts_abs = [], []

                    # POSITIVES — widely spread (component-wise) and scaled separation
                    if num_positive_points > 0:
                        pos_pts_roi = get_control_points(
                            roi_light,
                            scaled_min_blob,
                            num_positive_points,
                            min_sep=max(12, int(round(3 * scale_factor * erosion_kernel)))
                        )
                        pos_pts_abs = [[x1 + float(px), y1 + float(py)] for (px, py) in pos_pts_roi]

                    # NEGATIVES — inverse hues near the croc edges
                    if num_negative_points > 0:
                        # ring around positives within ROI
                        dil = cv2.dilate(roi_light, kernel, iterations=1)
                        ring = cv2.subtract(dil, roi_light)
                        ring_and_dark = cv2.bitwise_and(ring, roi_dark)
                        neg_pts_roi = []
                        if np.count_nonzero(ring_and_dark) > 0:
                            neg_pts_roi = get_control_points(
                                ring_and_dark, 3, num_negative_points,
                                min_sep=max(8, int(round(2 * scale_factor * erosion_kernel)))
                            )
                        # fallback: anywhere in inverse hue (but not croc hue)
                        if len(neg_pts_roi) < num_negative_points and np.count_nonzero(roi_dark) > 0:
                            more = get_control_points(
                                cv2.bitwise_and(roi_dark, cv2.bitwise_not(roi_light)),
                                scaled_min_blob, num_negative_points - len(neg_pts_roi),
                                min_sep=max(8, int(round(2 * scale_factor * erosion_kernel)))
                            )
                            neg_pts_roi += more

                        neg_pts_abs = [[x1 + float(nx), y1 + float(ny)] for (nx, ny) in neg_pts_roi]

                    all_pts = pos_pts_abs + neg_pts_abs
                    if all_pts:
                        pc_i = np.array(all_pts, dtype=np.float32)
                        pl_i = np.array([1] * len(pos_pts_abs) + [0] * len(neg_pts_abs), dtype=np.int64)
                        for (x, y), lbl in zip(pc_i, pl_i):
                            cv2.circle(debug_img_bgr, (int(x), int(y)), 5,
                                    (0, 255, 0) if int(lbl) == 1 else (0, 0, 255), -1)

                masks_i, scores, _ = predictor.predict(
                    point_coords=pc_i, point_labels=pl_i,
                    box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                    multimask_output=False
                )
                masks_i = _ensure_3d_mask(masks_i)  # (K,H,W)
                per_box_masks.append(masks_i)

            # ---- save preview ----
            debug_img_rgb = cv2.cvtColor(debug_img_bgr, cv2.COLOR_BGR2RGB)
            fn = f"{uuid.uuid4()}.png"
            Image.fromarray(debug_img_rgb).save(os.path.join(temp_path, fn))
            previews.append({"filename": fn, "subfolder": "", "type": "temp"})

            # ---- build outputs (IMPORTANT: pass RGBA) ----
            if per_box_masks:
                images_out, masks_out = create_tensor_output(image_np_rgba, per_box_masks, boxes)  # <- fixed
                res_images.extend(images_out)
                res_masks.extend(masks_out)

        if len(res_images) == 0:
            _, height, width, _ = image.shape
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return {"ui": {"images": []}, "result": (empty_mask, empty_mask)}

        return {"ui": {"images": previews}, "result": (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))}


