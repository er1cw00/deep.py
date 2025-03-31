import os
from app.base.error import Error

def get_mime_type_from_filepath(file_path):
  file_name = os.path.basename(file_path)
  file_root, file_ext = os.path.splitext(file_name) 
  ext_to_mime = {
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".png": "image/png",
      ".mp4": "video/mp4",
      ".m4v": "video/mp4",
      ".mov": "video/quicktime"
  }
  mime_type = ext_to_mime.get(file_ext.lower(), "none")
  return mime_type

def get_postfix_from_mime_type(mime_type):
    parts = mime_type.split("/")
    if len(parts) == 2:
        if parts[0] == 'image':
            if parts[1] == 'jpeg' or parts[1] == 'jpeg':
                return 'jpg', False, Error.OK
            elif parts[1] == 'png':
                return 'png', False, Error.OK
        elif parts[0] == 'video':
            if parts[1] == 'mp4' or parts[1] == 'm4v':
                return 'mp4', True, Error.OK
            elif parts[1] == 'quicktime':
                return 'mov', True, Error.OK
    return '', False, Error.UnsupportFile