from enum import Enum

class Error(Enum):
    OK = 'ok'
    Unknown = "unknown error"
    FileNotFound = 'file not found'
    TaskNotFount = 'task not found'
    MNotExist = 'dir not exist'
    NetworkError = 'network error'
    CredentialsError = 'credentials error'
    InvalidParameter = 'invalid parameter'
    UnsupportFile = 'unsupport file'
    FileSizeExceeded = 'file size exceeded'
    NoFaceDetected = 'no face detected'
    UnknownResponse = 'unknown response'
    SubprocessFail = 'subprocess fail'