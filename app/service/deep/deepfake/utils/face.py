import cv2
from collections import namedtuple
from typing import List, Literal

FaceAnalyserOrder = Literal['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small', 'best-worst', 'worst-best']

# Face = namedtuple('Face',
# [
# 	'bbox',
# 	'landmarks',
# 	'score',
# ])

import numpy as np
from numpy.linalg import norm as l2norm
#from easydict import EasyDict

class Face(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        #for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                    if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None
      
    @property
    def kps(self):
        return self.landmark_5
    
    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property 
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property 
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender==1 else 'F'

def expand_bbox(bbox, width, height, dsize=512):
	x1, y1, x2, y2 = map(int, bbox)

	# 计算中心点
	cx = (x1 + x2) // 2
	cy = (y1 + y2) // 2

	# 计算新的边界框
	new_x1 = max(0, cx - dsize // 2)
	new_y1 = max(0, cy - dsize // 2)
	new_x2 = min(width, cx + dsize // 2)
	new_y2 = min(height, cy + dsize // 2)

	# 如果因超出边界导致尺寸不足512x512，进行调整
	if new_x2 - new_x1 < dsize:
		if new_x1 == 0:
			new_x2 = min(width, new_x1 + dsize)
		else:
			new_x1 = max(0, new_x2 - dsize)

	if new_y2 - new_y1 < dsize:
		if new_y1 == 0:
			new_y2 = min(height, new_y1 + dsize)
		else:
			new_y1 = max(0, new_y2 - dsize)

	return (new_x1, new_y1, new_x2, new_y2)

def sort_by_order(faces : List[Face], order : FaceAnalyserOrder, face_center = None) -> List[Face]:
	if len(faces) <= 0:
		return faces
	if order == 'left-right':
		return sorted(faces, key = lambda face: face.bbox[0])
	if order == 'right-left':
		return sorted(faces, key = lambda face: face.bbox[0], reverse = True)
	if order == 'top-bottom':
		return sorted(faces, key = lambda face: face.bbox[1])
	if order == 'bottom-top':
		return sorted(faces, key = lambda face: face.bbox[1], reverse = True)
	if order == 'small-large':
		return sorted(faces, key = lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
	if order == 'large-small':
		return sorted(faces, key = lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse = True)
	if order == 'distance-from-retarget-face':
		return sorted(faces, key = lambda face: (((face.bbox[2]+face.bbox[0])/2-face_center[0])**2+((face.bbox[3]+face.bbox[1])/2-face_center[1])**2)**0.5)
	if order == 'best-worst':
		return sorted(faces, key = lambda face: face.score, reverse = True)
	if order == 'worst-best':
		return sorted(faces, key = lambda face: face.score)
	return faces


def draw_landmarks(frame, landmarks, color=(0, 255, 0), radius=2, thickness=-1):
    """
    在帧上绘制 203 个面部关键点。
    
    :param frame: 要绘制的图像 (numpy array)
    :param landmarks: 包含 203 个 (x, y) 坐标的列表或 numpy 数组
    :param color: 点的颜色，默认为绿色 (BGR)
    :param radius: 点的半径
    :param thickness: 线条厚度，-1 表示填充点
    """
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), radius, color, thickness)
    return frame