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
            if landmark_3d_68 and landmark_3d_68 and genderage and recognition:
                task_flags = {
                    'landmark_3d_68': landmark_3d_68,
                    'landmark_2d_106': landmark_2d_106,
                    'genderage': genderage,
                    'recognition': recognition
                }
                for taskname, model in self.models.items():
                    if task_flags.get(taskname):
                        print(f'Task enabled: {taskname}')
                        x = model.get(image, face)
                        print
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
    from deepfake.utils.face import draw_landmarks
    from deepfake.utils.video import get_video_writer
    from rich.progress import track
    
    def test_image(detector, input_path, output_path):
        image = cv2.imread(input_path)
        face_list = detector.get(image=image, landmark_3d_68=True, landmark_2d_106=True, genderage=True, recognition=True, order='best-worst')
        face = face_list[0]
        res = [512, 512]

        image = draw_landmarks(image, face.landmark_5)

        x1, y1, x2, y2 = map(int, face.bbox)
        face_crop = image[y1:y2, x1:x2]
        cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 1)
        resized_face = cv2.resize(face_crop, (512, 512))
        
        cv2.imwrite(output_path, image)
       
        
    insightface_path = "/Users/wadahana/workspace/AI/sd/ComfyUI/models/insightface"
    providers=['CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider']
    
    detector = FaceAnalysisDIY(name="buffalo_l", root=insightface_path, providers=providers)

    detector.prepare(ctx_id=0, det_size=(512, 512), det_thresh=0.5)
    detector.warmup()
    
    input_path = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250405/ec6ee635b4742b08e0fdea6c03769514/source.jpg"
    output_path = "/Users/wadahana/Desktop/output_face.jpg"    
    test_image(detector, input_path, output_path)
    
    print('test insightface finished! ')