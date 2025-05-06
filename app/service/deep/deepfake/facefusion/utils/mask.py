import cv2
import numpy as np

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

FaceMaskRegion = Literal['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip']

FaceMaskRegionMap : Dict[FaceMaskRegion, int] =\
{
    'skin': 1,
    'left-eyebrow': 2,
    'right-eyebrow': 3,
    'left-eye': 4,
    'right-eye': 5,
    'glasses': 6,
    'nose': 10,
    'mouth': 11,
    'upper-lip': 12,
    'lower-lip': 13
}

FaceMaskAllRegion = ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip']



def create_bbox_mask(crop_size, face_mask_blur, face_mask_padding):
    blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
    blur_area = max(blur_amount // 2, 1)
    box_mask = np.ones(crop_size, np.float32)
    box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
    box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
    box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
    box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
    if blur_amount > 0:
        box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
    return box_mask

def overlay_mask_on_face(face_img, mask, alpha=0.5, color=(0, 0, 255)):
    overlay = face_img.copy()
    color_layer = np.full_like(face_img, color, dtype=np.uint8)
    # 扩展 mask 到 3 通道（方便融合）
    mask_3ch = np.stack([mask]*3, axis=-1).astype(np.uint8)
    # 只对 mask 区域应用 alpha 混合
    overlay = np.where(mask_3ch > 0, cv2.addWeighted(color_layer, alpha, face_img, 1 - alpha, 0), face_img)
    
    return overlay