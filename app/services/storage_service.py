"""Storage service for MinIO/S3 file operations."""

import logging
from io import BytesIO
from datetime import timedelta

from minio import Minio
from minio.error import S3Error
from fastapi import UploadFile

from app.core.config import settings
from app.utils.file_utils import (
    validate_image_extension,
    validate_image_mimetype,
    validate_file_size,
    generate_unique_filename,
    sanitize_filename,
    get_file_mimetype,
)

logger = logging.getLogger(__name__)


class StorageService:
    """Service for MinIO/S3 storage operations."""

    def __init__(self):
        """Initialize MinIO client."""
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket_name = settings.minio_bucket_name
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """Ensure the bucket exists, create if it doesn't."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket exists: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Failed to ensure bucket exists: {e}")
            raise RuntimeError(f"MinIO bucket error: {e}") from e

    async def upload_file(
        self,
        file: UploadFile,
        folder: str = "",
        filename_prefix: str = "",
        validate_image: bool = True,
    ) -> dict:
        """
        Upload a file to MinIO storage.

        Args:
            file: FastAPI UploadFile object
            folder: Optional folder path in bucket (e.g., "faces", "violations")
            filename_prefix: Optional prefix for generated filename
            validate_image: Whether to validate as image file

        Returns:
            Dict with file info: {
                "filename": str,
                "object_name": str,
                "size": int,
                "content_type": str,
                "url": str
            }

        Raises:
            ValueError: If validation fails
            RuntimeError: If upload fails
        """
        # Validate filename
        if not file.filename:
            raise ValueError("Filename is required")

        # Validate image if required
        if validate_image:
            if not validate_image_extension(file.filename):
                raise ValueError(
                    "Invalid file extension. Allowed: .jpg, .jpeg, .png, .webp"
                )

            if file.content_type and not validate_image_mimetype(file.content_type):
                raise ValueError(
                    f"Invalid content type: {file.content_type}. "
                    f"Allowed: image/jpeg, image/png, image/webp"
                )

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Validate file size
        if not validate_file_size(file_size):
            raise ValueError(
                f"File size {file_size} bytes exceeds maximum allowed size"
            )

        # Generate unique filename
        sanitized = sanitize_filename(file.filename)
        unique_filename = generate_unique_filename(sanitized, prefix=filename_prefix)

        # Construct object name (path in bucket)
        if folder:
            object_name = f"{folder}/{unique_filename}"
        else:
            object_name = unique_filename

        # Determine content type
        content_type = (
            file.content_type
            or get_file_mimetype(file.filename)
            or "application/octet-stream"
        )

        try:
            # Upload to MinIO
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=BytesIO(content),
                length=file_size,
                content_type=content_type,
            )

            logger.info(f"Uploaded file: {object_name} ({file_size} bytes)")

            # Generate public URL (if bucket is public) or presigned URL
            url = self.generate_presigned_url(object_name)

            return {
                "filename": unique_filename,
                "object_name": object_name,
                "size": file_size,
                "content_type": content_type,
                "url": url,
            }

        except S3Error as e:
            logger.error(f"Failed to upload file {object_name}: {e}")
            raise RuntimeError(f"Failed to upload file: {e}") from e

    def download_file(self, object_name: str) -> bytes:
        """
        Download a file from MinIO storage.

        Args:
            object_name: Full object path in bucket

        Returns:
            File content as bytes

        Raises:
            RuntimeError: If download fails
        """
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            content = response.read()
            response.close()
            response.release_conn()

            logger.info(f"Downloaded file: {object_name}")
            return content

        except S3Error as e:
            logger.error(f"Failed to download file {object_name}: {e}")
            raise RuntimeError(f"Failed to download file: {e}") from e

    def delete_file(self, object_name: str) -> bool:
        """
        Delete a file from MinIO storage.

        Args:
            object_name: Full object path in bucket

        Returns:
            True if deletion was successful

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"Deleted file: {object_name}")
            return True

        except S3Error as e:
            logger.error(f"Failed to delete file {object_name}: {e}")
            raise RuntimeError(f"Failed to delete file: {e}") from e

    def generate_presigned_url(
        self,
        object_name: str,
        expiry: timedelta = timedelta(hours=24),
    ) -> str:
        """
        Generate a presigned URL for temporary access to a file.

        Args:
            object_name: Full object path in bucket
            expiry: URL expiration time (default: 24 hours)

        Returns:
            Presigned URL string

        Raises:
            RuntimeError: If URL generation fails
        """
        try:
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                expires=expiry,
            )
            logger.debug(f"Generated presigned URL for: {object_name}")
            return url

        except S3Error as e:
            logger.error(f"Failed to generate presigned URL for {object_name}: {e}")
            raise RuntimeError(f"Failed to generate presigned URL: {e}") from e

    def list_files(self, prefix: str = "", recursive: bool = True) -> list[dict]:
        """
        List files in bucket with optional prefix filter.

        Args:
            prefix: Filter by prefix (e.g., "faces/")
            recursive: List recursively

        Returns:
            List of file info dicts

        Raises:
            RuntimeError: If listing fails
        """
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=recursive,
            )

            files = []
            for obj in objects:
                files.append(
                    {
                        "object_name": obj.object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified,
                        "etag": obj.etag,
                    }
                )

            logger.info(f"Listed {len(files)} files with prefix: {prefix}")
            return files

        except S3Error as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            raise RuntimeError(f"Failed to list files: {e}") from e

    def file_exists(self, object_name: str) -> bool:
        """
        Check if a file exists in storage.

        Args:
            object_name: Full object path in bucket

        Returns:
            True if file exists
        """
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False


# Global storage service instance
storage_service = StorageService()
