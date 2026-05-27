"""
uav_augment.py — UAV-specific augmentations powered by albumentations.

Public entry points (same signatures as before, drop-in replacement):
  apply_weather_light_color(img, p=0.75)
      Apply stochastic weather / lighting / color augments to a raw BGR image.
      Call BEFORE letterbox.  Returns BGR uint8, same shape.

  bias_crop(img, labels, ...)
      Zoom-crop biased toward GT object clusters; albumentations handles all
      bbox coordinate transforms and filtering automatically.
      Call AFTER letterbox + pixel-xyxy label conversion, BEFORE random_affine.
      labels: N×6  [cls_id, track_id, x1, y1, x2, y2]  pixel coords.
      Returns (img, labels) with the same column layout.
"""

import random
import numpy as np
import albumentations as A

__all__ = ['apply_weather_light_color', 'bias_crop']

# ──────────────────────────────────────────────────────────────────────────────
# Weather / Lighting / Color pipeline  (built once, reused every call)
# ──────────────────────────────────────────────────────────────────────────────
# Design notes:
#   • Weather effects are mutually exclusive (OneOf) – at most one per sample.
#   • Lighting and color transforms are independent – can stack.
#   • Probabilities are tuned for VisDrone-style dense UAV data so that
#     augments are frequent enough to matter but not so aggressive that the
#     model never sees clean images.

_WEATHER_LIGHT_COLOR = A.Compose([

    # ── weather (pick at most one) ────────────────────────────────────────────
    A.OneOf([
        A.RandomFog(
            fog_coef_range=(0.20, 0.55),   # 20–55 % blend toward white veil
            alpha_coef=0.08,               # soft edge transition
            p=1.0,
        ),
        A.RandomRain(
            slant_range=(-12, 12),         # rain angle in degrees
            drop_length=25,                # pixel length per streak
            drop_width=1,
            drop_color=(200, 200, 200),    # near-white rain drops
            blur_value=3,                  # motion-blur kernel size
            brightness_coefficient=0.90,   # slight darkening during rain
            p=1.0,
        ),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),       # shadow can appear anywhere
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            shadow_intensity_range=(0.30, 0.70),
            p=1.0,
        ),
        A.RandomSnow(
            snow_point_range=(0.05, 0.20),
            brightness_coeff=1.5,
            p=1.0,
        ),
    ], p=0.55),   # 55 % chance of any weather effect

    # ── lighting ─────────────────────────────────────────────────────────────
    A.RandomBrightnessContrast(
        brightness_limit=(-0.35, 0.35),
        contrast_limit=(-0.35, 0.35),
        p=0.40,
    ),
    A.RandomGamma(
        gamma_limit=(40, 240),   # >100 → darker, <100 → brighter
        p=0.30,
    ),

    # ── color ─────────────────────────────────────────────────────────────────
    A.HueSaturationValue(
        hue_shift_limit=(-18, 18),
        sat_shift_limit=(-35, 35),
        val_shift_limit=(-25, 25),
        p=0.35,
    ),
    A.RGBShift(
        r_shift_limit=(-30, 30),
        g_shift_limit=(-30, 30),
        b_shift_limit=(-30, 30),
        p=0.25,
    ),
    A.ChannelShuffle(p=0.07),   # rare — strong appearance change

], p=1.0)


def apply_weather_light_color(img: np.ndarray, p: float = 0.75) -> np.ndarray:
    """
    Randomly augment a raw BGR image with weather / lighting / color effects.

    Parameters
    ----------
    img : H×W×3  BGR uint8  (raw image before letterbox)
    p   : master gate — probability of running any augmentation at all

    Returns
    -------
    Augmented BGR uint8 image, same shape as input.
    """
    if random.random() > p:
        return img
    # albumentations expects RGB; convert in/out
    rgb = img[:, :, ::-1]
    rgb = _WEATHER_LIGHT_COLOR(image=rgb)['image']
    return rgb[:, :, ::-1]


# ──────────────────────────────────────────────────────────────────────────────
# Bias Crop  (albumentations handles bbox math + filtering)
# ──────────────────────────────────────────────────────────────────────────────

def bias_crop(
    img: np.ndarray,
    labels: np.ndarray,
    crop_ratio_range: tuple = (0.65, 0.92),
    p_object: float = 0.80,
    min_visibility: float = 0.10,
) -> tuple:
    """
    Zoom-crop biased toward GT object clusters, then resize back to the
    original letterbox resolution.

    Albumentations handles:
      • Bbox coordinate rescaling after Crop + Resize
      • Clipping bboxes to image bounds  (clip=True)
      • Dropping boxes with < min_visibility visible area

    Parameters
    ----------
    img              : H×W×3 BGR uint8 — already letterboxed
    labels           : N×6 float32  [cls_id, track_id, x1, y1, x2, y2]
                       in pixel coords of the letterboxed image.
                       Pass np.array([]) if there are no labels.
    crop_ratio_range : (min, max) fraction of H / W for the crop window
    p_object         : probability of anchoring on a GT box (vs random anchor)
    min_visibility   : albumentations min_visibility threshold — boxes with
                       less than this fraction of their area visible after
                       the crop are discarded.

    Returns
    -------
    img_crop : H×W×3 BGR  — resized back to original letterbox resolution
    labels   : M×6 float32  (M ≤ N), same column layout as input
    """
    H, W = img.shape[:2]
    ratio = random.uniform(*crop_ratio_range)
    ch = max(2, int(H * ratio))
    cw = max(2, int(W * ratio))

    has_labels = len(labels) > 0

    # ── choose crop anchor ────────────────────────────────────────────────────
    if has_labels and random.random() < p_object:
        idx = random.randint(0, len(labels) - 1)
        cx = int((labels[idx, 2] + labels[idx, 4]) * 0.5)
        cy = int((labels[idx, 3] + labels[idx, 5]) * 0.5)
        # jitter: ±25 % of crop size around the chosen box centre
        cx += random.randint(-cw // 4, cw // 4)
        cy += random.randint(-ch // 4, ch // 4)
    else:
        cx = random.randint(cw // 2, max(cw // 2 + 1, W - cw // 2))
        cy = random.randint(ch // 2, max(ch // 2 + 1, H - ch // 2))

    x0 = int(max(0, min(W - cw, cx - cw // 2)))
    y0 = int(max(0, min(H - ch, cy - ch // 2)))
    x1_c = x0 + cw
    y1_c = y0 + ch

    # ── build albumentations pipeline for this crop ───────────────────────────
    crop_pipeline = A.Compose(
        [
            A.Crop(x_min=x0, y_min=y0, x_max=x1_c, y_max=y1_c),
            A.Resize(height=H, width=W),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',          # [x1, y1, x2, y2] pixel coords
            label_fields=['cls_ids', 'track_ids'],
            min_visibility=min_visibility,
            clip=True,                    # clip boxes to image bounds
        ),
    )

    # ── split label columns for albumentations ────────────────────────────────
    if has_labels:
        bboxes_in  = labels[:, 2:6].tolist()   # [x1, y1, x2, y2]
        cls_ids    = labels[:, 0].tolist()
        track_ids  = labels[:, 1].tolist()
    else:
        bboxes_in = []
        cls_ids   = []
        track_ids = []

    # albumentations expects RGB; we only care about pixel values for crop
    result = crop_pipeline(
        image=img,          # BGR is fine here — crop/resize are pixel-exact
        bboxes=bboxes_in,
        cls_ids=cls_ids,
        track_ids=track_ids,
    )

    img_out      = result['image']
    bboxes_out   = result['bboxes']
    cls_out      = result['cls_ids']
    track_out    = result['track_ids']

    # ── fallback: if all boxes were cropped away, keep original ──────────────
    if has_labels and len(bboxes_out) == 0:
        return img, labels

    # ── reassemble N×6 label array ────────────────────────────────────────────
    if len(bboxes_out) == 0:
        return img_out, labels   # no-label case

    labels_out = np.zeros((len(bboxes_out), 6), dtype=np.float32)
    labels_out[:, 0] = cls_out
    labels_out[:, 1] = track_out
    labels_out[:, 2:6] = np.array(bboxes_out, dtype=np.float32)

    return img_out, labels_out
