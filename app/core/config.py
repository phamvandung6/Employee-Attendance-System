"""Application configuration using Pydantic Settings."""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Application
    app_name: str = Field(default="Employee Attendance System")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="production")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/employee_attendance"
    )
    db_echo: bool = Field(default=False)

    # JWT Authentication
    secret_key: str = Field(default="your-secret-key-here")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)

    # Qdrant Vector Database
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")
    qdrant_collection_name: str = Field(default="face_embeddings")

    # MinIO/S3 Storage
    minio_endpoint: str = Field(default="localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_bucket_name: str = Field(default="employee-faces")
    minio_secure: bool = Field(default=False)

    # ML Models
    face_detection_model: str = Field(default="retinaface")
    face_recognition_model: str = Field(default="arcface")
    face_similarity_threshold: float = Field(default=0.6)
    model_path: str = Field(default="models/face_recognition")

    # API
    api_v1_prefix: str = Field(default="/api/v1")
    cors_origins: List[str] = Field(default=["http://localhost:3000"])

    # Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/app.log")


# Global settings instance
settings = Settings()
