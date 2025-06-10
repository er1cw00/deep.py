from enum import Enum

class Error(str, Enum):
    OK = 'ok'
    Unknown = "unknown error"
    FileNotFound = 'file not found'
    TaskNotFount = 'task not found'
    UnknownTask = 'unknown task'
    MNotExist = 'dir not exist'
    NetworkError = 'network error'
    CredentialsError = 'credentials error'
    InvalidParameter = 'invalid parameter'
    UnsupportFile = 'unsupport file'
    FileSizeExceeded = 'file size exceeded'
    UnknownResponse = 'unknown response'
    SubprocessFail = 'subprocess fail'
    FFmpegError = 'ffmpeg error'
    Unauthorized = 'unauthorized'
    NoFace = 'no face'
    NoAudio = 'no audio'