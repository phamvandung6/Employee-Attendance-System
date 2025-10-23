"""File utility functions for validation and processing."""

import mimetypes
from pathlib import Path
from typing import Optional
from datetime import datetime

# Allowed image formats
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}

# File size limits (in bytes)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB


def validate_image_extension(filename: str) -> bool:
    """
    Validate if file has allowed image extension.

    Args:
        filename: Name of the file to validate

    Returns:
        True if extension is allowed
    """
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


def validate_image_mimetype(content_type: str) -> bool:
    """
    Validate if content type is an allowed image mimetype.

    Args:
        content_type: MIME type of the file

    Returns:
        True if mimetype is allowed
    """
    return content_type in ALLOWED_IMAGE_MIMETYPES


def validate_file_size(file_size: int, max_size: int = MAX_IMAGE_SIZE) -> bool:
    """
    Validate if file size is within limits.

    Args:
        file_size: Size of file in bytes
        max_size: Maximum allowed size in bytes

    Returns:
        True if size is within limits
    """
    return 0 < file_size <= max_size


def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    """
    Generate unique filename with timestamp.

    Args:
        original_filename: Original file name
        prefix: Optional prefix for the filename

    Returns:
        Unique filename with timestamp
    """
    ext = Path(original_filename).suffix.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if prefix:
        return f"{prefix}_{timestamp}{ext}"
    return f"{timestamp}{ext}"


def get_file_mimetype(filename: str) -> Optional[str]:
    """
    Get MIME type for a filename.

    Args:
        filename: Name of the file

    Returns:
        MIME type string or None
    """
    mimetype, _ = mimetypes.guess_type(filename)
    return mimetype


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing potentially dangerous characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Keep only alphanumeric, dots, dashes, and underscores
    safe_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    sanitized = "".join(c if c in safe_chars else "_" for c in filename)

    # Ensure it doesn't start with a dot (hidden file)
    if sanitized.startswith("."):
        sanitized = "_" + sanitized[1:]

    return sanitized
