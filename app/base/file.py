import os
import filetype

def is_file(file_path : str) -> bool:
	return bool(file_path and os.path.isfile(file_path))


def is_directory(directory_path : str) -> bool:
	return bool(directory_path and os.path.isdir(directory_path))

def is_audio(audio_path : str) -> bool:
	return is_file(audio_path) and filetype.helpers.is_audio(audio_path)

def is_video(video_path : str) -> bool:
	return is_file(video_path) and filetype.helpers.is_video(video_path)

def is_image(image_path : str) -> bool:
	return is_file(image_path) and filetype.helpers.is_image(image_path)

