import os
import cv2
import time
import numpy as np
from rich.progress import track
from app.deepfake.utils.timer import Timer
from app.deepfake.utils.video import get_video_writer
from app.deepfake.utils.face import draw_landmarks, convert_face_landmark_68_to_5
from app.deepfake.facefusion.modules.yoloface import YoloFace
from app.deepfake.facefusion.modules.retinaface import RetinaFace
from app.deepfake.facefusion.modules.face_landmark import FaceLandmark_2dFan

from .file import get_test_files
    
def get_one_face(det, image):
    face_list = det.get(image=image, order='best-worst')
    if face_list != None and len(face_list) > 0:
        return face_list[0]
    return None

    
def test_image(det1, det2, landmark, input_path, output_path):
    t1 = Timer()
    t2 = Timer()
    t3 = Timer()
    
    image = cv2.imread(input_path)
    t1.tic()
    face1 = get_one_face(det1, image)
    t1.toc()
    
    t2.tic()
    face2 = get_one_face(det2, image)
    t2.toc()
    
    color = (0, 0, 200)
    color2 = (200,10,100)
    if face1 != None:
        t3.tic()
        face_landmark_68, face_landmark_68_score = landmark.get(image, face1.bbox)
        t3.toc()
        landmark_68_5 = convert_face_landmark_68_to_5(face_landmark_68)
        output1 = draw_landmarks(image.copy(), face1.landmark_5, color=color)
        output1 = draw_landmarks(output1, landmark_68_5, color=color2)
        x1, y1, x2, y2 = map(int, face1.bbox)
        cv2.rectangle(output1, (x1,y1), (x2,y2), color, 2)
        cv2.putText(output1, f'{face1.score:.4f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(output1, f'{face_landmark_68_score:.4f}', (x1,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,color2, 2)
    else:
        output1 = np.zeros_like(image)
        
    if face2 != None:
        t3.tic()
        face_landmark_68, face_landmark_68_score = landmark.get(image, face2.bbox)
        t3.toc()
        landmark_68_5 = convert_face_landmark_68_to_5(face_landmark_68)
        output2 = draw_landmarks(image.copy(), face2.landmark_5, color=color)  
        output2 = draw_landmarks(output2, landmark_68_5, color=color2)
        x1, y1, x2, y2 = map(int, face2.bbox)
        cv2.rectangle(output2, (x1,y1), (x2,y2), color, 2)
        cv2.putText(output2, f'{face2.score:.4f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(output2, f'{face_landmark_68_score:.4f}', (x1,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color2, 2)
    else:
        output2 = np.zeros_like(image)
        
    combined = cv2.hconcat([image, output1, output2])
    cv2.imwrite(output_path, combined)
    t1.show('yoloface')
    t2.show('retinaface')
    t3.show('landmark')
        
def test_video(det1, det2, landmark, input_path, output_path):
    t1 = Timer()
    t2 = Timer()
    t3 = Timer()
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
    print(f'video [{width}x{height}@{fps}] {total} frames!')
    
    writer = get_video_writer(output_path, fps)
    for i in track(range(total), description='Detecting....', transient=True):
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        t1.tic()
        face1 = get_one_face(det1, frame)
        t1.toc()
        
        t2.tic()
        face2 = get_one_face(det2, frame)
        t2.toc()
        
        color = (0, 0, 200)
        color2 = (200,10,100)
        if face1 == None:
            output1 = np.zeros_like(frame)
        else:
            t3.tic()
            face_landmark_68, face_landmark_68_score = landmark.get(frame, face1.bbox)
            t3.toc()
            landmark_68_5 = convert_face_landmark_68_to_5(face_landmark_68)
            output1 = draw_landmarks(frame.copy(), face1.landmark_5)
            output1 = draw_landmarks(output1, landmark_68_5, color=color2)
            x1, y1, x2, y2 = map(int, face1.bbox)
            cv2.rectangle(output1, (x1,y1), (x2,y2), color, 2)
            cv2.putText(output1, f'{face1.score:.4f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(output1, f'{face_landmark_68_score:.4f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color2, 2)
        if face2 == None:
            output2 = np.zeros_like(frame)
        else:
            t3.tic()
            face_landmark_68, face_landmark_68_score = landmark.get(frame, face2.bbox)
            t3.toc()
            landmark_68_5 = convert_face_landmark_68_to_5(face_landmark_68)
            output2 = draw_landmarks(frame.copy(), face2.landmark_5)
            output2 = draw_landmarks(output2, landmark_68_5, color=color2)
            x1, y1, x2, y2 = map(int, face2.bbox)
            cv2.rectangle(output2, (x1,y1), (x2,y2), color, 2)
            cv2.putText(output2, f'{face2.score:.4f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(output2, f'{face_landmark_68_score:.4f}', (x1,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color2, 2)
            
            combined = cv2.hconcat([frame, output1, output2])
            writer.append_data(combined[..., ::-1])

    cap.release()
    writer.close()
    
    t1.show('yoloface')
    t2.show('retinaface')
    t3.show('landmark')
    
retinaface_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/retinaface_10g.onnx'
#retinaface_path = "/Users/wadahana/workspace/AI/sd/ComfyUI/models/insightface"


yolo_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'
#yolo_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'

landmark_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/2dfan4.onnx'
#landmark_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/2dfan4.onnx'

providers=['CUDAExecutionProvider', 'CPUExecutionProvider', 'CoreMLExecutionProvider']

yoloface = YoloFace(model_path=yolo_path, providers=providers, threshold=0.5)
retinaface = RetinaFace(model_path=retinaface_path, providers=providers, threshold=0.5)
landmark = FaceLandmark_2dFan(model_path=landmark_path, providers=providers)


# input_path = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250405/ec6ee635b4742b08e0fdea6c03769514/source.jpg"
# output_path = "/Users/wadahana/Desktop/output_face.jpg"    
# test_image(yolo, insight, input_path=input_path, output_path=output_path)

# input_path = "/Users/wadahana/Desktop/sis/faceswap/test/mask/fbb6081fa3544ba51e4058b71660cfe3/target.mp4"
# output_path = "/Users/wadahana/Desktop/output_face.mp4"    
# test_video(yolo, insight, input_path=input_path, output_path=output_path)

photo_list, video_list = get_test_files()

print("Photo directories:")
for path in photo_list:
    input_path = os.path.join(path, 'target.jpg')
    output_path = os.path.join(path, 'target_face.jpg')
    print(path)
    test_image(yoloface, retinaface, landmark, input_path, output_path)
    test_image(yoloface, retinaface, landmark, os.path.join(path, 'source.jpg'), os.path.join(path, 'source_face.jpg'))

print("\nVideo directories:")
for path in video_list:
    input_path = os.path.join(path, 'target.mp4')
    output_path = os.path.join(path, 'target_face.mp4')
    print(path)
    test_video(yoloface, retinaface, landmark, input_path, output_path)
    test_image(yoloface, retinaface, landmark, os.path.join(path, 'source.jpg'), os.path.join(path, 'source_face.jpg'))
    time.sleep(10)
    
print('test face detect finished! ')


# 53dd886693270e1811a465740f7a266a
# 18cbf6266375bf4780b45bba72ed133a
# 87f97cd804122945562134319fa5d6ea
# fbb6081fa3544ba51e4058b71660cfe3