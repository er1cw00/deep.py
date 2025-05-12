# coding: utf-8

"""
face detectoin and alignment using InsightFace
"""

import numpy as np
from insightface.app import FaceAnalysis
#from insightface.app.common import Face
from deepfake.utils.timer import Timer
from deepfake.utils.face import Face, sort_by_order


class FaceAnalysisDIY(FaceAnalysis):
    def __init__(self, name='buffalo_l', root='~/.insightface', allowed_modules=None, **kwargs):
        super().__init__(name=name, root=root, allowed_modules=allowed_modules, **kwargs)

        self.timer = Timer()

    def get(self, image, **kwargs):
        #conf = kwargs.get('conf', 0.5)
        max_num = kwargs.get('max_face_num', 0)  # the number of the detected faces, 0 means no limit
        landmark_2d_106 = kwargs.get('landmark_2d_106', False)  # whether to do 106-point detection
        landmark_3d_68 = kwargs.get('landmark_3d_68', False)
        genderage = kwargs.get('genderage', False)
        recognition = kwargs.get('recognition', False)
        direction = kwargs.get('order', 'large-small')  # sorting order
        face_center = None

        bboxes, kpss = self.det_model.detect(image, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, landmark_5=kps, score=det_score)
            if landmark_2d_106 or landmark_3d_68 or genderage or recognition:
                task_flags = {
                    'landmark_3d_68': landmark_3d_68,
                    'landmark_2d_106': landmark_2d_106,
                    'genderage': genderage,
                    'recognition': recognition
                }
                for taskname, model in self.models.items():
                    if task_flags.get(taskname):
                        x = model.get(image, face)
                        print(f'{taskname} x: {x}')
            ret.append(face)

        ret = sort_by_order(ret, direction, face_center)
        return ret

    def warmup(self):
        self.timer.tic()

        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        self.get(img_bgr)

        elapse = self.timer.toc()
        print(f'FaceAnalysisDIY warmup time: {elapse:.3f}s')


if __name__ == "__main__":
    import cv2
    from deepfake.utils.face import convert_face_landmark_68_to_5, draw_landmarks
    from deepfake.utils.video import get_video_writer
    from rich.progress import track
    
    
    def test_image(detector, input_path, output_path):
        image = cv2.imread(input_path)
        t = Timer()
        t.tic()
        face_list = detector.get(image=image, landmark_3d_68=True, landmark_2d_106=False, genderage=False, recognition=False, order='best-worst')
        t.toc() 
        for face in face_list:
            landmark_5 = convert_face_landmark_68_to_5(face.landmark_3d_68[:, :2])
            image = draw_landmarks(image, face.landmark_3d_68[:, :2])
            x1, y1, x2, y2 = map(int, face.bbox)
            face_crop = image[y1:y2, x1:x2]
            cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 1)
            resized_face = cv2.resize(face_crop, (512, 512))
        
        cv2.imwrite(output_path, image)
        t.show('insightface detect photo')
        
    def test_video(detector, conf, input_path, output_path):
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
        t.show('insightface detect video')
            
    
    insightface_path = '/home/eric/workspace/AI/sd/ComfyUI/models/insightface'        
    #insightface_path = "/Users/wadahana/workspace/AI/sd/ComfyUI/models/insightface"
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider', 'CoreMLExecutionProvider']
    
    detector = FaceAnalysisDIY(name="buffalo_l", root=insightface_path, providers=providers)

    detector.prepare(ctx_id=0, det_size=(512, 512), det_thresh=0.4)
    detector.warmup()
    
    #ec6ee635b4742b08e0fdea6c03769514
    #input_path = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250405/0d4fc3a041d6befd7a2ee218da2d820b/target.jpg"
    #output_path = "/Users/wadahana/Desktop/output_face2.jpg"    
    input_path = "/home/eric/workspace/AI/sd/temp/mask/0d4fc3a041d6befd7a2ee218da2d820b/target.jpg"
    output_path = "/home/eric/workspace/AI/sd/temp/mask/0d4fc3a041d6befd7a2ee218da2d820b/output_face2.jpg" 

    test_image(detector, input_path, output_path)
    
    # input_path = "/home/eric/workspace/AI/sd/temp/mask/53dd886693270e1811a465740f7a266a/target.mp4"
    # output_path = "/home/eric/workspace/AI/sd/temp/mask/53dd886693270e1811a465740f7a266a/output_face3.mp4" 
    # test_video(detector, 0.4, input_path, output_path)
    
    
    # input_path = "/home/eric/workspace/AI/sd/temp/mask/87f97cd804122945562134319fa5d6ea/target.mp4"
    # output_path = "/home/eric/workspace/AI/sd/temp/mask/87f97cd804122945562134319fa5d6ea/output_face3.mp4" 
    # test_video(detector, 0.4, input_path, output_path)
    
    print('test insightface finished! ')