
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
    from urllib.parse import urlparse
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
        map_location="cpu"
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
    # boxes_filt not used here but kept for API parity
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
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    candidates = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_size:
            continue

        comp = (labels == lbl).astype(np.uint8)
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)

        k = max(3, 2 * int(min_sep) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        dt_dil = cv2.dilate(dt, kernel)

        maxima = (dt >= (dt_dil - 1e-6)) & (dt >= float(min_dt))
        ys, xs = np.where(maxima)
        vals = dt[ys, xs]

        for x, y, v in zip(xs, ys, vals):
            candidates.append((float(v), int(x), int(y)))

        if len(xs) == 0:
            yx = np.unravel_index(np.argmax(dt), dt.shape)
            y, x = int(yx[0]), int(yx[1])
            candidates.append((float(dt[y, x]), x, y))

    candidates.sort(key=lambda t: -t[0])

    chosen = []
    min_sep2 = float(min_sep) * float(min_sep)
    for _, x, y in candidates:
        ok = True
        for cx, cy in chosen:
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy < min_sep2:
                ok = False
                break
        if ok:
            chosen.append((x, y))
            if len(chosen) >= max_num:
                break

    return [[float(x), float(y)] for (x, y) in chosen]


def _ensure_3d_mask(m):
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    if m.ndim == 2:
        return m[None, :, :]
    elif m.ndim == 3:
        return m
    elif m.ndim == 4:
        return m[0, ...]
    else:
        raise ValueError(f"Unexpected mask ndim={m.ndim}")


def expand_box(x1, y1, x2, y2, W, H, pad_ratio=0.03):
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    return max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)


def get_control_points_balanced(mask_u8, total_k, min_sep=12, min_dt=0.5, min_area=1):
    mask_bin = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    comps = []
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comps.append((area, lbl))
    comps.sort(key=lambda t: -t[0])

    chosen = []
    for _, lbl in comps:
        comp = (labels == lbl).astype(np.uint8)
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        if dt.max() <= 0:
            continue
        y, x = np.unravel_index(np.argmax(dt), dt.shape)
        chosen.append([float(x), float(y)])
        if len(chosen) >= total_k:
            return chosen

    remaining = total_k - len(chosen)
    if remaining > 0:
        extra = get_control_points(mask_bin.astype(bool), min_size=min_area, max_num=remaining, min_sep=min_sep, min_dt=min_dt)

        def far_enough(nx, ny, lst, min_sep2):
            for cx, cy in lst:
                dx = nx - cx; dy = ny - cy
                if dx*dx + dy*dy < min_sep2:
                    return False
            return True

        ms2 = float(min_sep) * float(min_sep)
        for (nx, ny) in extra:
            if far_enough(nx, ny, chosen, ms2):
                chosen.append([nx, ny])
                if len(chosen) >= total_k:
                    break
    return chosen


def iou_bool(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-6)


def union_masks(mask_list):
    if not mask_list:
        return None
    out = mask_list[0].copy()
    for m in mask_list[1:]:
        out |= m
    return out


def hsv_mask_range(hsv, hmin_deg, hmax_deg, smin, vmin):
    """
    HSV mask for hue in degrees [0..360). OpenCV uses H in [0..179].
    Handles wrap-around if hmin > hmax.
    smin,vmin given in [0..1].
    Returns uint8 mask 0/255.
    """
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    hmin = (hmin_deg % 360) / 2.0
    hmax = (hmax_deg % 360) / 2.0
    smin255 = int(max(0.0, min(1.0, smin)) * 255)
    vmin255 = int(max(0.0, min(1.0, vmin)) * 255)

    if hmin <= hmax:
        lower = np.array([hmin, smin255, vmin255], dtype=np.float32)
        upper = np.array([hmax, 255, 255], dtype=np.float32)
        mask = cv2.inRange(hsv, lower, upper)
    else:
        # wrap-around: [hmin, 180] U [0, hmax]
        lower1 = np.array([hmin, smin255, vmin255], dtype=np.float32)
        upper1 = np.array([180, 255, 255], dtype=np.float32)
        lower2 = np.array([0, smin255, vmin255], dtype=np.float32)
        upper2 = np.array([hmax, 255, 255], dtype=np.float32)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                              cv2.inRange(hsv, lower2, upper2))
    return mask


def derive_negative_mask_from_pos(hsv, hmin_deg, hmax_deg, smin, vmin, gap_h, gap_s, gap_v):
    """
    Build a broad "likely-not-croc" mask by inverting the positive band with a gap.
    We consider pixels that are:
      - hue far outside [hmin,hmax] by at least gap_h (wrap-aware),
      - or have saturation < (smin - gap_s),
      - or value < (vmin - gap_v).
    The result is a 0/255 uint8 mask.
    """
    # Positive band
    pos_mask = hsv_mask_range(hsv, hmin_deg, hmax_deg, max(0, smin), max(0, vmin))

    # Gap thresholds
    s_cut = int(max(0.0, smin - gap_s) * 255)
    v_cut = int(max(0.0, vmin - gap_v) * 255)

    # Hue far-outside logic: create two expanded bands and invert
    # "Inside expanded" ~ [hmin-gap_h, hmax+gap_h]
    expanded = hsv_mask_range(hsv, hmin_deg - gap_h, hmax_deg + gap_h, 0.0, 0.0)
    hue_far = cv2.bitwise_not(expanded)  # hue outside the expanded band

    # Very desaturated or very dark also count as negatives
    desat_or_dark = cv2.bitwise_or((hsv[:, :, 1] < s_cut).astype(np.uint8) * 255,
                                   (hsv[:, :, 2] < v_cut).astype(np.uint8) * 255)

    neg_mask = cv2.bitwise_or(hue_far, desat_or_dark)

    # Never mark positive pixels as negatives
    neg_mask = cv2.bitwise_and(neg_mask, cv2.bitwise_not(pos_mask))
    return pos_mask, neg_mask


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
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "max_bounding_boxes": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),

                "enable_blob_detection": (["true", "false"], {"default": "true"}),

                # Positive croc band
                "pos_hue_min": ("FLOAT", {"default": 36.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "pos_hue_max": ("FLOAT", {"default": 132.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "pos_sat_min": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pos_val_min": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Derive negatives
                "auto_negative_from_pos": (["true", "false"], {"default": "true"}),
                "gap_hue": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 60.0, "step": 1.0}),
                "gap_sat": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "gap_val": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),

                # Manual negatives (used only if auto_negative_from_pos == "false")
                "neg_hue_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "neg_hue_max": ("FLOAT", {"default": 360.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "neg_sat_max": ("FLOAT", {"default": 0.07, "min": 0.0, "max": 1.0, "step": 0.01}),
                "neg_val_max": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),

                "min_blob_size": ("INT", {"default": 300, "min": 10, "max": 10000, "step": 10}),
                "num_positive_points": ("INT", {"default": 8, "min": 0, "max": 20, "step": 1}),
                "num_negative_points": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
                "erosion_kernel": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),

                # Safety: turn SAM multimask off by default; we enable in fallback to reduce peak memory
                "use_multimask_first": (["false", "true"], {"default": "false"}),
            }
        }

    CATEGORY = "segment_anything2"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt, threshold, max_bounding_boxes,
             enable_blob_detection, pos_hue_min, pos_hue_max, pos_sat_min, pos_val_min,
             auto_negative_from_pos, gap_hue, gap_sat, gap_val,
             neg_hue_min, neg_hue_max, neg_sat_max, neg_val_max,
             min_blob_size, num_positive_points, num_negative_points, erosion_kernel,
             use_multimask_first):

        res_images = []
        res_masks = []
        previews = []
        temp_path = folder_paths.get_temp_directory()

        for item in image:
            item = Image.fromarray(
                np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)
            ).convert("RGBA")
            boxes = groundingdino_predict(grounding_dino_model, item, prompt, threshold)
            if boxes.shape[0] == 0:
                continue

            # Limit boxes
            if boxes.shape[0] > max_bounding_boxes:
                boxes = boxes[:max_bounding_boxes]

            img_np = np.array(item)[:, :, :3]  # RGB
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

            # Build positive & negative masks (full-image) once
            pos_mask_full, neg_mask_full = None, None
            if auto_negative_from_pos == "true":
                pos_mask_full, neg_mask_full = derive_negative_mask_from_pos(
                    hsv, pos_hue_min, pos_hue_max, pos_sat_min, pos_val_min, gap_hue, gap_sat, gap_val
                )
            else:
                pos_mask_full = hsv_mask_range(hsv, pos_hue_min, pos_hue_max, pos_sat_min, pos_val_min)
                # manual negatives: anything with hue in [neg_hue_min,neg_hue_max] and low S/V
                hmin = (neg_hue_min % 360) / 2.0
                hmax = (neg_hue_max % 360) / 2.0
                smax255 = int(max(0.0, min(1.0, neg_sat_max)) * 255)
                vmax255 = int(max(0.0, min(1.0, neg_val_max)) * 255)
                if hmin <= hmax:
                    lower = np.array([hmin, 0, 0], dtype=np.float32)
                    upper = np.array([hmax, smax255, vmax255], dtype=np.float32)
                    neg_mask_full = cv2.inRange(hsv, lower, upper)
                else:
                    lower1 = np.array([hmin, 0, 0], dtype=np.float32)
                    upper1 = np.array([180, smax255, vmax255], dtype=np.float32)
                    lower2 = np.array([0, 0, 0], dtype=np.float32)
                    upper2 = np.array([hmax, smax255, vmax255], dtype=np.float32)
                    neg_mask_full = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                                                   cv2.inRange(hsv, lower2, upper2))
                neg_mask_full = cv2.bitwise_and(neg_mask_full, cv2.bitwise_not(pos_mask_full))

            kernel = np.ones((max(1, erosion_kernel), max(1, erosion_kernel)), np.uint8)

            # Debug image
            debug_img = np.array(item)[:, :, :3]
            debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            boxes_np = boxes.numpy().astype(int)
            for box in boxes_np:
                x1, y1, x2, y2 = box
                cv2.rectangle(debug_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            predictor = SAM2ImagePredictor(sam_model)
            image_np = np.array(item)
            image_np_rgb = image_np[..., :3]

            # SAM embedding (most memory-heavy stage)
            try:
                predictor.set_image(image_np_rgb)
            except Exception as e:
                # Surface SAM errors rather than crash the process
                logger.error(f"SAM set_image failed: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            per_box_masks = []
            H, W = image_np_rgb.shape[:2]
            union_bool_masks = []

            for bi in range(boxes_np.shape[0]):
                x1, y1, x2, y2 = [int(v) for v in boxes_np[bi]]
                x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, W, H, pad_ratio=0.03)
                if x2 <= x1 or y2 <= y1:
                    continue

                # ROI masks
                pos_roi = pos_mask_full[y1:y2, x1:x2]
                neg_roi_full = neg_mask_full[y1:y2, x1:x2]

                # Positives: balanced seeding
                pos_pts_roi = []
                if enable_blob_detection == "true" and num_positive_points > 0:
                    box_area = (x2 - x1) * (y2 - y1)
                    min_blob_px = max(50, min_blob_size, int(0.0005 * box_area))
                    pos_pts_roi = get_control_points_balanced(
                        pos_roi, total_k=num_positive_points,
                        min_sep=max(12, 3*erosion_kernel), min_dt=0.3, min_area=min_blob_px
                    )

                # Negatives: ring around positive within box, intersected with neg_roi_full
                neg_pts_roi = []
                if enable_blob_detection == "true" and num_negative_points > 0:
                    dil = cv2.dilate(pos_roi, kernel, iterations=1)
                    ring = cv2.subtract(dil, pos_roi)
                    ring = cv2.bitwise_and(ring, cv2.bitwise_not(pos_roi))
                    ring = cv2.bitwise_and(ring, neg_roi_full)
                    neg_pts_roi = get_control_points(
                        ring, min_size=3, max_num=num_negative_points, min_sep=max(8, 2*erosion_kernel), min_dt=0.0
                    )
                    # fallback if ring not enough
                    if len(neg_pts_roi) < num_negative_points:
                        more = get_control_points(
                            cv2.bitwise_and(neg_roi_full, cv2.bitwise_not(pos_roi)),
                            min_size=3, max_num=(num_negative_points - len(neg_pts_roi)),
                            min_sep=max(8, 2*erosion_kernel), min_dt=0.0
                        )
                        neg_pts_roi += more

                pos_abs = [[x1 + float(px), y1 + float(py)] for (px, py) in pos_pts_roi]
                neg_abs = [[x1 + float(nx), y1 + float(ny)] for (nx, ny) in neg_pts_roi]
                all_pts = pos_abs + neg_abs
                pc_i = np.array(all_pts, dtype=np.float32) if all_pts else None
                pl_i = np.array([1]*len(pos_abs) + [0]*len(neg_abs), dtype=np.int64) if all_pts else None

                # Draw debug points
                if all_pts:
                    for (x, y), lbl in zip(pc_i, pl_i):
                        color = (0, 255, 0) if int(lbl) == 1 else (0, 0, 255)
                        cv2.circle(debug_img_bgr, (int(x), int(y)), 5, color, -1)

                # First try: single mask to minimize memory
                try:
                    masks_i, scores, _ = predictor.predict(
                        point_coords=pc_i, point_labels=pl_i,
                        box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                        multimask_output=(use_multimask_first == "true")
                    )
                except RuntimeError as re:
                    logger.error(f"SAM predict OOM/RuntimeError: {re}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # try once more with fewer points
                    try:
                        fewer_pc = None
                        fewer_pl = None
                        if pc_i is not None and pc_i.shape[0] > 4:
                            fewer_pc = pc_i[:4, :]
                            fewer_pl = pl_i[:4]
                        masks_i, scores, _ = predictor.predict(
                            point_coords=fewer_pc, point_labels=fewer_pl,
                            box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                            multimask_output=False
                        )
                    except Exception as e2:
                        logger.error(f"SAM predict failed again: {e2}")
                        continue
                except Exception as e:
                    logger.error(f"SAM predict failed: {e}")
                    continue

                masks_i = _ensure_3d_mask(masks_i)  # (K,H,W)

                # If we asked for single mask, evaluate coverage; if poor, do a multimask refinement
                best = None
                best_iou = -1.0
                pos_roi_bool = (pos_roi > 0)

                if masks_i.shape[0] == 1 and np.any(pos_roi_bool):
                    mk = masks_i[0].astype(bool)
                    iou = iou_bool(mk[y1:y2, x1:x2], pos_roi_bool)
                    best = mk
                    best_iou = iou
                    # fallback pass: try multimask and pick best if IoU is weak
                    if iou < 0.70:
                        try:
                            m2, s2, _ = predictor.predict(
                                point_coords=pc_i, point_labels=pl_i,
                                box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                                multimask_output=True
                            )
                            m2 = _ensure_3d_mask(m2)
                            for k in range(m2.shape[0]):
                                mk2 = m2[k].astype(bool)
                                i2 = iou_bool(mk2[y1:y2, x1:x2], pos_roi_bool)
                                if i2 > best_iou:
                                    best_iou = i2
                                    best = mk2
                        except Exception as e:
                            logger.warning(f"Multimask refinement failed: {e}")
                else:
                    # Already multimask: pick best by IoU
                    for k in range(masks_i.shape[0]):
                        mk = masks_i[k].astype(bool)
                        iou = iou_bool(mk[y1:y2, x1:x2], pos_roi_bool) if np.any(pos_roi_bool) else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best = mk

                if best is None:
                    continue

                # Coverage boost
                covered = np.logical_and(best[y1:y2, x1:x2], pos_roi_bool).sum()
                total_ref = pos_roi_bool.sum()
                coverage = float(covered) / float(total_ref + 1e-6) if total_ref > 0 else 0.0

                if enable_blob_detection == "true" and coverage < 0.80 and num_positive_points >= 4 and np.any(pos_roi_bool):
                    uncovered = cv2.bitwise_and(pos_roi_bool.astype(np.uint8)*255,
                                                cv2.bitwise_not(best[y1:y2, x1:x2].astype(np.uint8))*255)
                    extra_pts_roi = get_control_points_balanced(
                        uncovered, total_k=min(3, num_positive_points),
                        min_sep=max(12, 3*erosion_kernel), min_dt=0.0, min_area=30
                    )
                    if extra_pts_roi:
                        extra_abs = [[x1 + float(px), y1 + float(py)] for (px, py) in extra_pts_roi]
                        pc2 = np.array(extra_abs, dtype=np.float32) if pc_i is None else np.concatenate([pc_i, np.array(extra_abs, dtype=np.float32)], axis=0)
                        pl2 = np.ones(len(extra_pts_roi), dtype=np.int64) if pl_i is None else np.concatenate([pl_i, np.ones(len(extra_pts_roi), dtype=np.int64)], axis=0)
                        try:
                            m3, s3, _ = predictor.predict(
                                point_coords=pc2, point_labels=pl2,
                                box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                                multimask_output=True
                            )
                            m3 = _ensure_3d_mask(m3)
                            for k in range(m3.shape[0]):
                                mk3 = m3[k].astype(bool)
                                i3 = iou_bool(mk3[y1:y2, x1:x2], pos_roi_bool)
                                if i3 > best_iou:
                                    best_iou = i3
                                    best = mk3
                        except Exception as e:
                            logger.warning(f"Coverage refinement failed: {e}")

                per_box_masks.append(best[None, ...])
                union_bool_masks.append(best)

            # Save preview
            debug_img_rgb = cv2.cvtColor(debug_img_bgr, cv2.COLOR_BGR2RGB)
            try:
                fn = f"{uuid.uuid4()}.png"
                full_path = os.path.join(temp_path, fn)
                Image.fromarray(debug_img_rgb).save(full_path)
                previews.append({"filename": fn, "subfolder": "", "type": "temp"})
            except Exception as e:
                logger.warning(f"Failed saving preview: {e}")

            final_bool = union_masks(union_bool_masks)
            if final_bool is not None:
                per_box_masks = [final_bool[None, ...].astype(np.float32)]

            if len(per_box_masks) > 0:
                images, masks = create_tensor_output(image_np, per_box_masks, boxes)
                res_images.extend(images)
                res_masks.extend(masks)

        if len(res_images) == 0:
            _, height, width, _ = image.shape
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return {"ui": {"images": []}, "result": (empty_mask, empty_mask)}
        return {"ui": {"images": previews}, "result": (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))}
