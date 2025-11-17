"""
Custom exceptions for the FaceRelis application
"""


class FaceRelisException(Exception):
    """Base exception for all FaceRelis-related errors"""
    pass


class FaceDetectionError(FaceRelisException):
    """Raised when face detection fails"""
    pass


class ClusteringError(FaceRelisException):
    """Raised when clustering operation fails"""
    pass


class FileOperationError(FaceRelisException):
    """Raised when file system operations fail"""
    pass


class ConfigurationError(FaceRelisException):
    """Raised when configuration is invalid"""
    pass


class ValidationError(FaceRelisException):
    """Raised when input validation fails"""
    pass


class TaskError(FaceRelisException):
    """Raised when background task operations fail"""
    pass


class ModelLoadError(FaceRelisException):
    """Raised when ML model loading fails"""
    pass
