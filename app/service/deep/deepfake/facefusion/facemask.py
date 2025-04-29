import cv2
import os
import numpy as np

from .modules.xseg import XSeg
from .modules.resnet34 import Resnet34
from .utils.mask import create_bbox_mask, FaceMaskRegion, FaceMaskRegionMap, FaceMaskAllRegion
from dataclasses import dataclass, field 
from typing import Literal, List, Tuple
from liveportrait.config.base_config import PrintableConfig


@dataclass(repr=False)  # use repr from PrintableConfig
class FaceMaskConfig(PrintableConfig):
    providers:  List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    model_path: str = ""
    bbox: bool = True
    occlusion: bool = False
    region: bool = False
    bbox_blur: float = 0.3
    bbox_padding: Tuple[int, int, int, int] = (0,0,0,0)
    region_list: List[FaceMaskRegion] = field( default_factory=lambda: FaceMaskAllRegion.copy() )


class FaceMasker(object):
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.occlusion == True:
            self.xseg = XSeg(os.path.join(self.cfg.model_path, 'dfl_xseg.onnx'), self.cfg.providers)
        if self.cfg.region == True:
            self.resnet = Resnet34(os.path.join(self.cfg.model_path, 'bisenet_resnet_34.onnx'), self.cfg.providers)
    
    def create_mask(self, crop_face):
        mask_list = []
        if self.cfg.bbox == True:
            bbox_mask = create_bbox_mask(crop_face.shape[:2][::-1], self.cfg.bbox_blur, self.cfg.bbox_padding)
            mask_list.append(bbox_mask)
        if self.cfg.occlusion == True:
            occlusion_mask = self.xseg.detect(image=crop_face)
            mask_list.append(occlusion_mask)
        if self.cfg.region == True:
            region_mask = self.resnet.detect(crop_face, self.cfg.region_list)
            mask_list.append(region_mask)
        crop_mask = np.minimum.reduce(mask_list).clip(0, 1)  
        return crop_mask
        
    def create_bbox_mask(self, crop_size, face_mask_blur, face_mask_padding):
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
    


if __name__ == "__main__":
    from .yoloface import YoloFace
    from rich.progress import track
    from ..utils.affine import arcface_128_v2, ffhq_512, warp_face_by_landmark, paste_back, blend_frame
    
    import imageio
    import cv2
    import time
    import os
        
    def get_video_writer(outout_path, fps):
            video_format = 'mp4'     # default is mp4 format
            codec = 'libx265'        # default is libx264 encoding
            #quality = quality        # video quality
            pixelformat = 'yuv420p'  # video pixel format
            image_mode = 'rbg'
            macro_block_size = 2
            ffmpeg_params = ['-crf', '22', '-preset', 'medium', '-tag:v', 'hvc1']
            writer = imageio.get_writer(uri=outout_path,
                                format=video_format,
                                fps=fps, 
                                codec=codec, 
                                #quality=quality, 
                                ffmpeg_params=ffmpeg_params, 
                                pixelformat=pixelformat, 
                                macro_block_size=macro_block_size)
            return writer