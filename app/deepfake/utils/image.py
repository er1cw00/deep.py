
from PIL import Image

def constrain_image(image, max_width, max_height, min_width, min_height, crop_if_required):
    current_width, current_height = image.size
    aspect_ratio = current_width / current_height

    constrained_width = max(min(current_width, min_width), max_width)
    constrained_height = max(min(current_height, min_height), max_height)

    if constrained_width / constrained_height > aspect_ratio:
        constrained_width = max(int(constrained_height * aspect_ratio), min_width)
        if crop_if_required:
            constrained_height = int(current_height / (current_width / constrained_width))
    else:
        constrained_height = max(int(constrained_width / aspect_ratio), min_height)
        if crop_if_required:
            constrained_width = int(current_width / (current_height / constrained_height))
    
    if constrained_width == current_width and constrained_height == current_height: 
        return image
    
    resized_image = image.resize((constrained_width, constrained_height), Image.LANCZOS)
    
    if crop_if_required and (constrained_width > max_width or constrained_height > max_height):
        left = max((constrained_width - max_width) // 2, 0)
        top = max((constrained_height - max_height) // 2, 0)
        right = min(constrained_width, max_width) + left
        bottom = min(constrained_height, max_height) + top
        resized_image = resized_image.crop((left, top, right, bottom))
        
    return resized_image

def check_resolution(width=1024, height=1024):
    def check_size(side):
        side = max(512, min(4096, side))   # 先限制在范围内
        side = (side // 8) * 8              # 再向下对齐到最近的8的倍数
        return side
    return check_size(width), check_size(height)