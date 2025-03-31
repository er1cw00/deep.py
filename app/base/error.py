from enum import Enum

class Error(Enum):
    OK = 'ok'
    Unknown = "unknown error"
    FileNotFound = 'file not found'
    TaskNotFount = 'task not found'
    NetworkError = 'network error'
    CredentialsError = 'credentials error'
    InvalidParameter = 'invalid parameter'
    UnsupportFile = 'unsupport file'
    FileSizeExceeded = 'file size exceeded'