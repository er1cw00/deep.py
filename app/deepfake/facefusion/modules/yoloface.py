import os
import cv2
import onnx
import onnxruntime
import numpy as np
from collections import namedtuple
from typing import List
from app.deepfake.utils.face import Face, sort_by_order, resize_frame_resolution, expand_bounding_box, apply_nms
 
class YoloFace:
    def __init__(self, model_path, providers, threshold=0.5):
        self.threshold = threshold
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        print(f'YoloFace providers:{providers}; current providers: {self.session.get_providers()}') 
        inputs = self.session.get_inputs()
        self.input_size = (inputs[0].shape[2], inputs[0].shape[3])
        self.input_name = inputs[0].name
            
    def pre_process(self, image):
        img = np.zeros((self.input_size[0], self.input_size[1], 3))
        img[:image.shape[0], :image.shape[1], :] = image
        img = (img - 127.5) / 128.0
        img = np.expand_dims(img.transpose(2, 0, 1), axis = 0).astype(np.float32)
        return img
        
    def post_process(self, size, bounding_box_list, face_landmark_5_list, score_list):
        sort_indices = np.argsort(-np.array(score_list))
        bounding_box_list = [ bounding_box_list[index] for index in sort_indices ]
        face_landmark_5_list = [face_landmark_5_list[index] for index in sort_indices]
        score_list = [ score_list[index] for index in sort_indices ]
        
        face_list = []
        keep_indices = apply_nms(bounding_box_list, 0.4)
        for index in keep_indices:
            bounding_box = bounding_box_list[index]
            face_landmark = face_landmark_5_list[index]
            score = score_list[index]
            face_list.append(Face(bbox=expand_bounding_box(size, bounding_box),
                                    landmark_5=face_landmark, 
                                    score=score))
        return face_list
            

    def get(self, image, order='left-right'):
        img = resize_frame_resolution(image, self.input_size)
        detect_img = self.pre_process(img)
        ratio_height = image.shape[0] / img.shape[0]
        ratio_width = image.shape[1] / img.shape[1]
        
        outputs = self.session.run(None, {self.input_name: detect_img})
        outputs = np.squeeze(outputs).T
        bounding_box_raw, score_raw, face_landmark_5_raw = np.split(outputs, [ 4, 5 ], axis = 1)
        bounding_box_list = []
        face_landmark_5_list = []
        score_list = []
        keep_indices = np.where(score_raw > self.threshold)[0]
        if keep_indices.any():
            bounding_box_raw, face_landmark_5_raw, score_raw = bounding_box_raw[keep_indices], face_landmark_5_raw[keep_indices], score_raw[keep_indices]
            for bounding_box in bounding_box_raw:
                bounding_box_list.append(np.array(
                [
                    (bounding_box[0] - bounding_box[2] / 2) * ratio_width,
                    (bounding_box[1] - bounding_box[3] / 2) * ratio_height,
                    (bounding_box[0] + bounding_box[2] / 2) * ratio_width,
                    (bounding_box[1] + bounding_box[3] / 2) * ratio_height
                ]))
            face_landmark_5_raw[:, 0::3] = (face_landmark_5_raw[:, 0::3]) * ratio_width
            face_landmark_5_raw[:, 1::3] = (face_landmark_5_raw[:, 1::3]) * ratio_height
            for face_landmark_5 in face_landmark_5_raw:
                face_landmark_5_list.append(np.array(face_landmark_5.reshape(-1, 3)[:, :2]))
            score_list = score_raw.ravel().tolist()
        
        faces = self.post_process(image.shape, bounding_box_list, face_landmark_5_list, score_list)
        if len(faces) > 1:
            faces = sort_by_order(faces, order)
        return faces 
    

if __name__ == "__main__":
    from app.deepfake.utils.face import draw_landmarks
    from app.deepfake.utils.video import get_video_writer
    from app.deepfake.utils.timer import Timer
    from rich.progress import track
    
    def test_image(detector, input_path, output_path):
        #input = '/Users/wadahana/Desktop/ad_enhance-d77b92ad.png'
        #input = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250405/ec6ee635b4742b08e0fdea6c03769514/source.jpg"
        
        image = cv2.imread(input_path)
        t = Timer()
        t.tic()
        face_list = detector.get(image=image,  order='best-worst')
        t.toc()
        for face in face_list:
            image = draw_landmarks(image, face.landmark_5)
            color = (200,10,200)
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(image, (x1,y1), (x2,y2), color, 1)
            cv2.putText(image, f'{face.score:.4f}', (x1,y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        #face_crop = image[y1:y2, x1:x2]
        #resized_face = cv2.resize(face_crop, (512, 512))
        
        cv2.imwrite(output_path, image)
        t.show('yolo face detect photo')
        


    def test_video(detector, input_path, output_path):
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        t = Timer()

        color = (200,10,200)
        writer = get_video_writer(output_path=output_path, fps=fps)
        
        for i in track(range(total), description='Detecting....', transient=True):
            ret, frame = cap.read()
            if not ret:
                break
            t.tic()
            face_list = detector.get(image=frame)
            t.toc()

            for face in face_list:
                frame = draw_landmarks(frame, face.landmark_5)
                x1, y1, x2, y2 = map(int, face.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                cv2.putText(frame, f'{face.score:.4f}', (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            #(x1, y1, x2, y2) = adjust_bounding_box(bbox=face.bounding_box, width=width, height=height, dsize=512)
            #face_crop = frame[y1:y2, x1:x2]
            #resized_face = cv2.resize(face_crop, (512, 512))
            writer.append_data(frame[..., ::-1])
    
        writer.close
        cap.release()
        t.show('yolo face detect video')


    #model_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'
    model_path = "/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx"
    providers=['CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider']
   
    yolo = YoloFace(model_path=model_path, providers=providers, threshold=0.5)
    input_path = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250405/ec6ee635b4742b08e0fdea6c03769514/source.jpg"
    output_path = "/Users/wadahana/Desktop/output_face.jpg"    
    # input_path = "/home/eric/workspace/AI/sd/temp/mask/2b4194098d22b327b28893378e8a6c99/target.jpg"
    # output_path = "/home/eric/workspace/AI/sd/temp/mask/2b4194098d22b327b28893378e8a6c99/output_face2.jpg"   
    test_image(yolo, input_path, output_path)
    
    # input_path = "/home/eric/workspace/AI/sd/temp/mask/53dd886693270e1811a465740f7a266a/target.mp4"
    # output_path = "/home/eric/workspace/AI/sd/temp/mask/53dd886693270e1811a465740f7a266a/output_face2.mp4" 
    # test_video(yolo, input_path, output_path)
    
    
    # input_path = "/home/eric/workspace/AI/sd/temp/mask/87f97cd804122945562134319fa5d6ea/target.mp4"
    # output_path = "/home/eric/workspace/AI/sd/temp/mask/87f97cd804122945562134319fa5d6ea/output_face2.mp4" 
    # test_video(yolo,  input_path, output_path)
    
    input_path = "/Users/wadahana/Desktop/sis/faceswap/test/mask/fbb6081fa3544ba51e4058b71660cfe3/target.mp4"
    output_path = "/Users/wadahana/Desktop/sis/faceswap/test/mask/fbb6081fa3544ba51e4058b71660cfe3/output_face2.mp4" 
    test_video(yolo,  input_path, output_path)
    
    print('test yoloface_onnx finished! ')
