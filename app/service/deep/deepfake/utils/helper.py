import cv2

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