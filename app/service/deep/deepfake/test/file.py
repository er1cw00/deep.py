import os

MASK_DIR = '/home/eric/workspace/AI/sd/temp/mask'

def get_test_files():
    photo_list = []
    video_list = []

    for subdir in os.listdir(MASK_DIR):
        subdir_path = os.path.join(MASK_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue

        target_jpg = os.path.join(subdir_path, 'target.jpg')
        target_mp4 = os.path.join(subdir_path, 'target.mp4')

        if os.path.isfile(target_jpg):
            photo_list.append(subdir_path)
        elif os.path.isfile(target_mp4):
            video_list.append(subdir_path)
    return photo_list, video_list