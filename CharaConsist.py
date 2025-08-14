# Port of CharaConsist-style consistency to SDXL/UNet for ComfyUI
# Implements:
#   (1) Point-tracking attention (identity K/V reindexed to matched coords + concat) on self-attn
#   (2) Adaptive token merge (decaying alpha with match-confidence)
#   (3) Decoupled foreground/background gating, optional background sharing
#
# Notes for SDXL port:
# - We only modify SELF-ATTENTION calls (context is None or same token length as x).
# - DiT RPE is emulated by reindexing identity tokens to matched frame positions prior to concat.
# - Identity caches are populated via a one-time warm-up call inside the sampler (using the provided identity image latent).
#
# Installation: drop into ComfyUI/custom_nodes and restart ComfyUI.

import math
import types
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Utilities: find CrossAttention-like modules ----------
def _iter_cross_attn_modules(unet: nn.Module):
    for name, mod in unet.named_modules():
        has_qkv = all(hasattr(mod, a) for a in ["to_q", "to_k", "to_v"])
        has_attn = hasattr(mod, "forward") and callable(mod.forward)
        if has_qkv and has_attn:
            yield name, mod


def _get_parent_module(root: nn.Module, name: str) -> nn.Module:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent


# ---------- CharaConsist state ----------
class CCState:
    def __init__(self, device="cuda"):
        self.device = device
        # Identity caches (per layer)
        self.id_k: Dict[str, torch.Tensor] = {}
        self.id_v: Dict[str, torch.Tensor] = {}
        self.id_attn_out: Dict[str, torch.Tensor] = {}

        # Point matches (per layer)
        self.pt_idx: Dict[str, torch.Tensor] = {}
        self.pt_conf: Dict[str, torch.Tensor] = {}

        # Token masks (per layer)
        self.fg_mask_tokens: Dict[str, torch.Tensor] = {}
        self.bg_mask_tokens: Dict[str, torch.Tensor] = {}

        # Cached gating mask M per layer (vectorized bool mask)
        self.gating_M: Dict[str, torch.Tensor] = {}

        # Identity latent (B,4,H/8,W/8) for the warm-up call
        self.identity_latent: Optional[torch.Tensor] = None

        # Config
        self.share_bg: bool = False
        self.use_adaptive_merge: bool = True
        self.merge_alpha0: float = 0.5
        self.apply_from_frac: float = 0.22
        self.apply_until_frac: float = 0.80
        self.layers_filter: List[str] = []  # substrings to select layers

        # Runtime
        self.total_steps: int = 50
        self.current_step: int = 0
        self.mode: str = "frame"  # "identity" or "frame"
        self.enabled: bool = False

        # Optional: manual mask at base res (1x1xH x W or HxW), values in [0,1]
        self.manual_mask: Optional[torch.Tensor] = None

        # One-time warnings
        self._warned_b_gt1: bool = False
        self._warned_no_id_latent: bool = False

    def alpha_at_step(self) -> float:
        s, S = self.current_step, self.total_steps
        s0 = int(self.apply_from_frac * S)
        s1 = int(self.apply_until_frac * S)
        if s <= s0:
            return self.merge_alpha0
        if s >= s1:
            return 0.0
        return float(self.merge_alpha0 * (1.0 - (s - s0) / max(1, (s1 - s0))))

    def clear_runtime(self):
        self.id_k.clear(); self.id_v.clear(); self.id_attn_out.clear()
        self.pt_idx.clear(); self.pt_conf.clear()
        self.fg_mask_tokens.clear(); self.bg_mask_tokens.clear()
        self.gating_M.clear()
        self.identity_latent = None
        self._warned_b_gt1 = False
        self._warned_no_id_latent = False


# ---------- Matching ----------
def cosine_top1_match(frame_feats: torch.Tensor, id_feats: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    # frame_feats: [Nf, C], id_feats: [Ni, C], both normalized
    sim = frame_feats @ id_feats.t()                # [Nf, Ni]
    conf, idx = torch.max(sim, dim=1)               # [Nf]
    conf = (conf.clamp(-1, 1) + 1.0) * 0.5          # [-1,1] -> [0,1]
    return idx, conf


# ---------- CrossAttention wrapper ----------
class CCWrappedCrossAttn(nn.Module):
    def __init__(self, base_attn: nn.Module, ccstate: CCState, layer_name: str):
        super().__init__()
        self.base = base_attn
        self.ccs = ccstate
        self.layer = layer_name
        self.orig_forward = base_attn.forward

    def _project_mask_to_tokens(self, x, qh):
        # Lazily project manual mask to this layer's token grid once
        if (self.layer not in self.ccs.fg_mask_tokens) and (self.ccs.manual_mask is not None):
            N = qh.shape[2]
            s = int(math.isqrt(N))
            if s * s == N:
                m = self.ccs.manual_mask
                if m.dim() == 2:
                    m = m.unsqueeze(0).unsqueeze(0)
                elif m.dim() == 3:
                    m = m.unsqueeze(1)
                m = m.to(x.device, dtype=torch.float32)
                m_small = F.interpolate(m, size=(s, s), mode="nearest")
                fg = (m_small > 0.5).view(1, N)       # [1, N]
                self.ccs.fg_mask_tokens[self.layer] = fg
                self.ccs.bg_mask_tokens[self.layer] = (~fg).to(fg.device)

    def forward(self, x, context=None, **kwargs):
        ccs = self.ccs
        if (not ccs.enabled) or (ccs.mode not in ["identity", "frame"]):
            return self.orig_forward(x, context=context, **kwargs)

        # Only patch selected layers (if filter set)
        if ccs.layers_filter:
            if not any(k in self.layer for k in ccs.layers_filter):
                return self.orig_forward(x, context=context, **kwargs)

        # Only manipulate SELF-ATTN; skip pure cross-attn to text
        is_self = (context is None) or (context.shape[1] == x.shape[1] and context.shape[-1] == x.shape[-1])
        if not is_self:
            return self.orig_forward(x, context=context, **kwargs)

        # Extract Q/K/V
        h = self.base.heads if hasattr(self.base, "heads") else 8
        to_q, to_k, to_v, to_out = self.base.to_q, self.base.to_k, self.base.to_v, self.base.to_out

        q_in = x
        k_in = x if context is None else context  # for self-attn, context==x

        q = to_q(q_in); k = to_k(k_in); v = to_v(k_in)

        def rearr(t):
            B, N, C = t.shape
            dh = C // h
            return t.view(B, N, h, dh).permute(0, 2, 1, 3)

        qh = rearr(q)            # [B,H,Nf,Dh]
        kh = rearr(k)
        vh = rearr(v)
        B, H, Nf, Dh = qh.shape

        if B != 1 and not ccs._warned_b_gt1:
            print("[CharaConsist][warn] Batched generation (B>1) detected. The SDXL port "
                  "currently shares the foreground mask across batch and derives point-matches from batch 0.")
            ccs._warned_b_gt1 = True

        # Identity pass: cache id_k/id_v/id_attn_out (full B kept)
        if ccs.mode == "identity":
            scale = Dh ** -0.5
            attn_scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
            attn_weights = attn_scores.softmax(dim=-1)
            out = torch.matmul(attn_weights, vh)      # [B,H,Nf,Dh]
            self.ccs.id_k[self.layer] = kh.detach()
            self.ccs.id_v[self.layer] = vh.detach()
            self.ccs.id_attn_out[self.layer] = out.detach()
            out = out.permute(0, 2, 1, 3).contiguous().view(q.shape)
            return to_out(out)

        # Frame pass: outside the apply window, run vanilla attention
        s, S = ccs.current_step, ccs.total_steps
        s0 = int(ccs.apply_from_frac * S)
        s1 = int(ccs.apply_until_frac * S)
        if s < s0 or s > s1 or self.layer not in self.ccs.id_k:
            scale = Dh ** -0.5
            attn_scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
            attn_weights = attn_scores.softmax(dim=-1)
            out = torch.matmul(attn_weights, vh)
            out = out.permute(0, 2, 1, 3).contiguous().view(q.shape)
            return to_out(out)

        # Ensure per-layer fg/bg tokens are available if manual_mask provided
        self._project_mask_to_tokens(x, qh)

        # ---- Point-tracking attention ----
        # Prepare features for matching (once at s == s0) — we derive matches from batch 0.
        key_for_match = kh.mean(dim=1)[0]        # [Nf,Dh] (batch 0)
        id_k = self.ccs.id_k[self.layer]         # [B,H,Ni,Dh]
        id_key_for_match = id_k.mean(dim=1)[0]   # [Ni,Dh] (batch 0)
        f_feats = F.normalize(key_for_match, dim=-1)
        id_feats = F.normalize(id_key_for_match, dim=-1)

        if (self.layer not in self.ccs.pt_idx) or (s == s0):
            # Foreground selection from layer mask (if any)
            if self.layer in self.ccs.fg_mask_tokens:
                fg_mask_flat = self.ccs.fg_mask_tokens[self.layer].to(x.device)[0]  # [Nf]
                frame_fg_idx = torch.nonzero(fg_mask_flat).squeeze(1)
            else:
                frame_fg_idx = torch.arange(Nf, device=x.device)

            f_feats_sel = f_feats[frame_fg_idx]
            idx, conf = cosine_top1_match(f_feats_sel, id_feats)
            self.ccs.pt_idx[self.layer]  = idx.detach()
            self.ccs.pt_conf[self.layer] = conf.detach()
        else:
            if self.layer in self.ccs.fg_mask_tokens:
                fg_mask_flat = self.ccs.fg_mask_tokens[self.layer].to(x.device)[0]
                frame_fg_idx = torch.nonzero(fg_mask_flat).squeeze(1)
            else:
                frame_fg_idx = torch.arange(Nf, device=x.device)
            idx  = self.ccs.pt_idx[self.layer]
            conf = self.ccs.pt_conf[self.layer]

        # Reindex identity K/V to matched coord positions (per-batch pad, shared indices across batch)
        id_k_flat_all = id_k            # [B,H,Ni,Dh]
        id_v_flat_all = self.ccs.id_v[self.layer]  # [B,H,Ni,Dh]
        pad_k_list = []
        pad_v_list = []
        for b in range(B):
            id_k_flat = id_k_flat_all[b]
            id_v_flat = id_v_flat_all[b]
            id_k_fg   = id_k_flat[:, idx, :]          # [H,N_fg,Dh]
            id_v_fg   = id_v_flat[:, idx, :]          # [H,N_fg,Dh]
            pad_k_b = torch.zeros_like(kh[b])         # [H,Nf,Dh]
            pad_v_b = torch.zeros_like(vh[b])
            pad_k_b[:, frame_fg_idx, :] = id_k_fg
            pad_v_b[:, frame_fg_idx, :] = id_v_fg

            # Optionally map background positions (share_bg=True)
            if self.ccs.share_bg and (self.layer in self.ccs.fg_mask_tokens):
                fg = self.ccs.fg_mask_tokens[self.layer].to(x.device)[0]
                bg_idx = torch.nonzero(~fg).squeeze(1)
                if bg_idx.numel() > 0:
                    sim_bg, id_bg_idx = torch.max(f_feats[bg_idx] @ id_feats.t(), dim=1)
                    pad_k_b[:, bg_idx, :] = id_k_flat[:, id_bg_idx, :]
                    pad_v_b[:, bg_idx, :] = id_v_flat[:, id_bg_idx, :]

            pad_k_list.append(pad_k_b.unsqueeze(0))
            pad_v_list.append(pad_v_b.unsqueeze(0))

        id_k_concat = torch.cat(pad_k_list, dim=0)  # [B,H,Nf,Dh]
        id_v_concat = torch.cat(pad_v_list, dim=0)  # [B,H,Nf,Dh]
        k_aug = torch.cat([kh, id_k_concat], dim=2)   # [B,H,Nf+Nf,Dh]
        v_aug = torch.cat([vh, id_v_concat], dim=2)

        # ---- FG/BG gating mask (vectorized, cached per layer) ----
        scale = Dh ** -0.5
        attn_scores = torch.matmul(qh, k_aug.transpose(-2, -1)) * scale   # [B,H,Nf,Nf+Nf]

        if self.layer in self.ccs.gating_M:
            M = self.ccs.gating_M[self.layer]
        else:
            M = torch.ones((Nf, k_aug.shape[2]), dtype=torch.bool, device=x.device)
            left = Nf
            if self.layer in self.ccs.fg_mask_tokens:
                fg = self.ccs.fg_mask_tokens[self.layer].to(x.device)[0]      # [Nf]
                fg_idx = torch.nonzero(fg).squeeze(1)
                bg_idx = torch.nonzero(~fg).squeeze(1)
                if fg_idx.numel() > 0 and bg_idx.numel() > 0:
                    # forbid frame_fg -> id_bg
                    M[fg_idx.unsqueeze(1), left + bg_idx.unsqueeze(0)] = False
                    if self.ccs.share_bg:
                        # forbid frame_bg -> id_fg
                        M[bg_idx.unsqueeze(1), left + fg_idx.unsqueeze(0)] = False
            self.ccs.gating_M[self.layer] = M

        attn_scores = attn_scores.masked_fill(~M.unsqueeze(0).unsqueeze(1), float("-inf"))
        attn_weights = attn_scores.softmax(dim=-1)

        # ---- Adaptive token merge ----
        out_aug = torch.matmul(attn_weights, v_aug)  # [B,H,Nf,Dh]
        if self.ccs.use_adaptive_merge and (self.layer in self.ccs.id_attn_out):
            alpha = self.ccs.alpha_at_step()
            if alpha > 0.0:
                id_out_all = self.ccs.id_attn_out[self.layer]  # [B,H,Ni,Dh]
                # Use batch-specific identity outputs but shared indices
                pad_out_list = []
                for b in range(B):
                    id_out_flat = id_out_all[b]                 # [H,Ni,Dh]
                    id_out_fg   = id_out_flat[:, idx, :]        # [H,N_fg,Dh]
                    pad_out_b   = torch.zeros_like(out_aug[b])  # [H,Nf,Dh]
                    pad_out_b[:, frame_fg_idx, :] = id_out_fg
                    pad_out_list.append(pad_out_b.unsqueeze(0))
                pad_out = torch.cat(pad_out_list, dim=0)        # [B,H,Nf,Dh]

                conf_pad = torch.zeros((B, Nf), device=x.device)
                conf_pad[:, frame_fg_idx] = self.ccs.pt_conf[self.layer].to(x.device)  # broadcast same conf to all B
                conf_pad = conf_pad.clamp(0, 1)

                out_aug = (1.0 - alpha) * out_aug + alpha * conf_pad.unsqueeze(1).unsqueeze(-1) * pad_out

        out = out_aug.permute(0, 2, 1, 3).contiguous().view(q.shape)
        return to_out(out)


# ---------- Patch / Unpatch ----------
def patch_unet_with_cc(unet: nn.Module, ccstate: CCState) -> Dict[str, Any]:
    originals = {}
    for name, mod in _iter_cross_attn_modules(unet):
        parent = _get_parent_module(unet, name)
        leaf = name.split(".")[-1]
        originals[name] = getattr(parent, leaf)
        setattr(parent, leaf, CCWrappedCrossAttn(mod, ccstate, layer_name=name))
    return originals


def unpatch_unet(unet: nn.Module, originals: Dict[str, Any]):
    for name, orig in originals.items():
        parent = _get_parent_module(unet, name)
        leaf = name.split(".")[-1]
        setattr(parent, leaf, orig)


# ---------- ComfyUI nodes ----------
class CC_SDXL_IdentityCache:
    """
    Configures CharaConsist state and stores the identity latent.
    The actual identity caching (id_k/id_v/id_attn_out) is performed by a one-time warm-up call
    inside the patched sampler (apply_model wrapper), guaranteeing correct shapes.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "identity_image": ("IMAGE",),
            },
            "optional": {
                "manual_mask": ("MASK",),  # 1=foreground
                "apply_from_frac": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0}),
                "apply_until_frac": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0}),
                "merge_alpha0": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "use_adaptive_merge": ("BOOL", {"default": True}),
                "share_bg": ("BOOL", {"default": False}),
                "layers_filter": ("STRING", {"default": "middle_block, output_blocks"}),  # comma-separated substrings
            }
        }
    RETURN_TYPES = ("CC_STATE",)
    FUNCTION = "build"
    CATEGORY = "CharaConsist/SDXL"

    def build(self, model, identity_image, manual_mask=None, apply_from_frac=0.22, apply_until_frac=0.80,
              merge_alpha0=0.5, use_adaptive_merge=True, share_bg=False, layers_filter="middle_block, output_blocks"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ccs = CCState(device=device)
        ccs.enabled = True
        ccs.mode = "identity"  # will switch to 'frame' after warm-up
        ccs.apply_from_frac = float(apply_from_frac)
        ccs.apply_until_frac = float(apply_until_frac)
        ccs.merge_alpha0 = float(merge_alpha0)
        ccs.use_adaptive_merge = bool(use_adaptive_merge)
        ccs.share_bg = bool(share_bg)
        ccs.layers_filter = [s.strip() for s in layers_filter.split(",")] if layers_filter else []

        # Encode identity image to latent for the warm-up pass
        vae = model.model_vae if hasattr(model, "model_vae") else model.first_stage_model
        img = identity_image.to(device=device, dtype=torch.float32)
        img = img * 2.0 - 1.0
        with torch.no_grad():
            z = vae.encode(img)[0].to(device)
        ccs.identity_latent = z

        # Stash manual mask if provided
        if manual_mask is not None:
            m = manual_mask
            if m.dim() == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.dim() == 3:
                m = m.unsqueeze(1)
            ccs.manual_mask = m.to(device=device, dtype=torch.float32).clamp(0, 1)

        return (ccs,)


class CC_SDXL_AttnPatch:
    """
    Patches the model for CharaConsist behavior. Connect output to KSampler.model.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cc_state": ("CC_STATE",),
                "total_steps": ("INT", {"default": 16, "min": 1, "max": 200}),
                "layers_filter": ("STRING", {"default": "middle_block, output_blocks"}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "CharaConsist/SDXL"

    def patch(self, model, cc_state, total_steps=16, layers_filter="middle_block, output_blocks"):
        unet = model.model.diffusion_model if hasattr(model, "model") else model.diffusion_model
        ccs: CCState = cc_state
        ccs.enabled = True
        ccs.total_steps = int(total_steps)
        if layers_filter:
            ccs.layers_filter = [s.strip() for s in layers_filter.split(",")]

        # Patch attention modules
        model._cc_original_attn = patch_unet_with_cc(unet, ccs)

        # Wrap apply_model for one-time identity warm-up & safe step counting
        if not hasattr(model, "_cc_original_apply_model"):
            model._cc_original_apply_model = model.apply_model

            def wrapped_apply_model(self_model, x, t, c, *args, **kwargs):
                # One-time identity warm-up to populate id_k/id_v/id_attn_out with correct shapes
                if not getattr(self_model, "_cc_did_identity", False):
                    if ccs.identity_latent is not None:
                        prev_mode = ccs.mode
                        ccs.mode = "identity"
                        try:
                            id_x = ccs.identity_latent
                            # align to current spatial size, dtype, device
                            if id_x.shape[-2:] != x.shape[-2:]:
                                id_x = F.interpolate(id_x, size=x.shape[-2:], mode="bilinear", align_corners=False)
                            if (id_x.dtype != x.dtype) or (id_x.device != x.device):
                                id_x = id_x.to(device=x.device, dtype=x.dtype)
                            _ = self_model._cc_original_apply_model(id_x, t, c, *args, **kwargs)
                        finally:
                            ccs.mode = prev_mode if prev_mode in ("identity", "frame") else "frame"
                            self_model._cc_did_identity = True
                    else:
                        if not ccs._warned_no_id_latent:
                            print("[CharaConsist][warn] identity_latent missing; skipping identity warm-up. "
                                  "Consistency will be inactive until caches are populated.")
                            ccs._warned_no_id_latent = True
                        self_model._cc_did_identity = True

                # Step counting: increment only when t changes
                t_scalar = float(t.detach().mean().item()) if torch.is_tensor(t) else float(t)
                last_t = getattr(self_model, "_cc_last_t", None)
                if (last_t is None) or (abs(t_scalar - last_t) > 1e-12):
                    self_model._cc_last_t = t_scalar
                    step = getattr(self_model, "_cc_step", 0) + 1
                    self_model._cc_step = min(step, ccs.total_steps - 1)
                ccs.current_step = getattr(self_model, "_cc_step", 0)
                ccs.mode = "frame"
                return self_model._cc_original_apply_model(x, t, c, *args, **kwargs)

            model.apply_model = types.MethodType(wrapped_apply_model, model)

        return (model,)


class CC_SDXL_Unpatch:
    """
    Restores the model and clears CC caches.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "cc_state": ("CC_STATE",),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "unpatch"
    CATEGORY = "CharaConsist/SDXL"

    def unpatch(self, model, cc_state=None):
        unet = model.model.diffusion_model if hasattr(model, "model") else model.diffusion_model
        if hasattr(model, "_cc_original_attn"):
            unpatch_unet(unet, model._cc_original_attn)
            del model._cc_original_attn
        # restore apply_model
        for attr in ("_cc_original_apply_model", "_cc_step", "_cc_last_t", "_cc_did_identity"):
            if hasattr(model, attr):
                if attr == "_cc_original_apply_model":
                    model.apply_model = getattr(model, attr)
                delattr(model, attr)
        # clear runtime caches
        if isinstance(cc_state, CCState):
            cc_state.clear_runtime()
        return (model,)


NODE_CLASS_MAPPINGS = {
    "CC_SDXL_IdentityCache": CC_SDXL_IdentityCache,
    "CC_SDXL_AttnPatch":     CC_SDXL_AttnPatch,
    "CC_SDXL_Unpatch":       CC_SDXL_Unpatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CC_SDXL_IdentityCache": "CharaConsist SDXL: Identity Cache",
    "CC_SDXL_AttnPatch":     "CharaConsist SDXL: Patch Attention",
    "CC_SDXL_Unpatch":       "CharaConsist SDXL: Unpatch",
}
