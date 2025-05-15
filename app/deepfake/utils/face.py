import cv2
import numpy as np
from functools import lru_cache
from typing import List, Literal, Any

FaceAnalyserOrder = Literal['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small', 'best-worst', 'worst-best']

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
        if self._normed_embedding is None:
            if self.embedding is None:
                return None
            self._normed_embedding = self.embedding / self.embedding_norm
        return self._normed_embedding

    @property 
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender==1 else 'F'

def resize_frame_resolution(vision_frame , max_resolution):
    height, width = vision_frame.shape[:2]
    max_width, max_height = max_resolution
    if height > max_height or width > max_width:
        scale = min(max_height / height, max_width / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(vision_frame, (new_width, new_height))
    return vision_frame


def convert_face_landmark_68_to_5(landmark_68):
	landmark_5 = np.array(
	[
		np.mean(landmark_68[36:42], axis = 0),
		np.mean(landmark_68[42:48], axis = 0),
		landmark_68[30],
		landmark_68[48],
		landmark_68[54]
	])
	return landmark_5

@lru_cache(maxsize = None)
def create_static_anchors(feature_stride : int, anchor_total : int, stride_height : int, stride_width : int) -> np.ndarray[Any, Any]:
	y, x = np.mgrid[:stride_height, :stride_width][::-1]
	anchors = np.stack((y, x), axis = -1)
	anchors = (anchors * feature_stride).reshape((-1, 2))
	anchors = np.stack([ anchors ] * anchor_total, axis = 1).reshape((-1, 2))
	return anchors

def distance_to_bounding_box(points : np.ndarray[Any, Any], distance : np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
	x1 = points[:, 0] - distance[:, 0]
	y1 = points[:, 1] - distance[:, 1]
	x2 = points[:, 0] + distance[:, 2]
	y2 = points[:, 1] + distance[:, 3]
	bounding_box = np.column_stack([ x1, y1, x2, y2 ])
	return bounding_box


def distance_to_face_landmark_5(points : np.ndarray[Any, Any], distance : np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
	x = points[:, 0::2] + distance[:, 0::2]
	y = points[:, 1::2] + distance[:, 1::2]
	face_landmark_5 = np.stack((x, y), axis = -1)
	return face_landmark_5

def apply_nms(bounding_box_list, iou_threshold):
    keep_indices = []
    dimension_list = np.reshape(bounding_box_list, (-1, 4))
    x1 = dimension_list[:, 0]
    y1 = dimension_list[:, 1]
    x2 = dimension_list[:, 2]
    y2 = dimension_list[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.arange(len(bounding_box_list))
    while indices.size > 0:
        index = indices[0]
        remain_indices = indices[1:]
        keep_indices.append(index)
        xx1 = np.maximum(x1[index], x1[remain_indices])
        yy1 = np.maximum(y1[index], y1[remain_indices])
        xx2 = np.minimum(x2[index], x2[remain_indices])
        yy2 = np.minimum(y2[index], y2[remain_indices])
        width = np.maximum(0, xx2 - xx1 + 1)
        height = np.maximum(0, yy2 - yy1 + 1)
        iou = width * height / (areas[index] + areas[remain_indices] - width * height)
        indices = indices[np.where(iou <= iou_threshold)[0] + 1]
    return keep_indices

def expand_bounding_box(size, bbox, expansion=32):
    """
    根据给定的 bbox，在四个边各扩展一定像素。
    
    :param image: 输入的图像
    :param bbox: 原始边界框，格式为 (x1, y1, x2, y2)
    :param expansion: 每个边扩展的像素数
    :return: 扩展后的图像
    """
    x1, y1, x2, y2 = bbox
    # 扩展边界框
    x1_expanded = max(x1 - expansion, 0)  # 保证不越界
    y1_expanded = max(y1 - expansion, 0)
    x2_expanded = min(x2 + expansion, size[1])  # 保证不越界
    y2_expanded = min(y2 + expansion, size[0])
    return [x1_expanded, y1_expanded, x2_expanded, y2_expanded]

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



# def expand_bbox(bbox, width, height, dsize=512):
# 	x1, y1, x2, y2 = map(int, bbox)

# 	# 计算中心点
# 	cx = (x1 + x2) // 2
# 	cy = (y1 + y2) // 2

# 	# 计算新的边界框
# 	new_x1 = max(0, cx - dsize // 2)
# 	new_y1 = max(0, cy - dsize // 2)
# 	new_x2 = min(width, cx + dsize // 2)
# 	new_y2 = min(height, cy + dsize // 2)

# 	# 如果因超出边界导致尺寸不足512x512，进行调整
# 	if new_x2 - new_x1 < dsize:
# 		if new_x1 == 0:
# 			new_x2 = min(width, new_x1 + dsize)
# 		else:
# 			new_x1 = max(0, new_x2 - dsize)

# 	if new_y2 - new_y1 < dsize:
# 		if new_y1 == 0:
# 			new_y2 = min(height, new_y1 + dsize)
# 		else:
# 			new_y1 = max(0, new_y2 - dsize)

# 	return (new_x1, new_y1, new_x2, new_y2)