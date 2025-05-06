import cv2
from rich.progress import track
from deepfake.utils.helper import draw_landmarks
from deepfake.facefusion.modules.yoloface import YoloFace
from deepfake.facefusion.modules.face_analysis_diy import FaceAnalysisDIY
#from live_portrait.utils.video import images2video

    
def test_image(detector):
    #input = '/Users/wadahana/Desktop/ad_enhance-d77b92ad.png'
    input = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250405/ec6ee635b4742b08e0fdea6c03769514/source.jpg"
    image = cv2.imread(input)
    face_list = detector.detect(image=image, conf=0.5, order='best-worst')
    face = face_list[0]
    res = [512, 512]
    #box_mask = create_box_mask(res, 0.3, (0,0,0,0))
    #crop_mask = np.minimum.reduce([box_mask]).clip(0, 1)
    
    image = draw_landmarks(image, face[1])

    x1, y1, x2, y2 = map(int, face[0])
    face_crop = image[y1:y2, x1:x2]
    cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 1)
    resized_face = cv2.resize(face_crop, (512, 512))
    cv2.imwrite('/Users/wadahana/Desktop/output.jpg', image)
    #cv2.imwrite('/Users/wadahana/Desktop/output_mask_png', crop_mask)
        
    def adjust_bounding_box(bbox, width, height, dsize=512):
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

    def test_video(detector):
        from facefusion.utils.affine import warp_face_by_landmark, paste_back
        #video_input = '../assets/dzq.mp4'
        
        video_input = '/Users/wadahana/Desktop/sis/faceswap/test/sq/suck2/suck2-short.mp4'
        cap = cv2.VideoCapture(video_input)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
        frames = []
        #while True:
        for i in track(range(total), description='Detecting....', transient=True):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face_list = detector.detect(image=frame, conf=0.7)
            if len(face_list) == 0:
                continue
            face = face_list[0]
            frame = draw_landmarks(frame, face[1])
            x1, y1, x2, y2 = map(int, face[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 

            #(x1, y1, x2, y2) = adjust_bounding_box(bbox=face.bounding_box, width=width, height=height, dsize=512)
            face_crop = frame[y1:y2, x1:x2]
            resized_face = cv2.resize(face_crop, (512, 512))
            #frames.append(resized_face)
            #out.write(resized_face)
            
            frames.append(frame)
    
        images2video(frames, wfp='../output_yoloface.mp4', fps=fps)
        cap.release()



    model_path = '../../../models/facefusion/yoloface_8n.onnx'
    providers=['CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider']
   
    detector = YoloFace(model_path=model_path, providers=providers)
    test_image(detector)
    print('test yoloface_onnx finished! ')
