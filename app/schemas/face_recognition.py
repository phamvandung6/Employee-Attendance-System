"""Face recognition schemas."""

from datetime import datetime
from pydantic import BaseModel, Field


class FaceRegistrationRequest(BaseModel):
    """Request schema for face registration."""

    employee_id: int = Field(..., description="Employee ID to register face for")
    check_duplicate: bool = Field(
        True, description="Check if face is already registered"
    )


class FaceRegistrationResponse(BaseModel):
    """Response schema for face registration."""

    success: bool
    employee_id: int
    point_id: str | None = None
    processing_time: float
    message: str
    error: str | None = None


class FaceMatchInfo(BaseModel):
    """Information about a face match."""

    employee_id: int | None
    confidence: float
    is_match: bool


class FaceRecognitionResponse(BaseModel):
    """Response schema for face recognition."""

    success: bool
    processing_time: float
    faces_detected: int
    faces_matched: int
    matches: list[FaceMatchInfo]
    error: str | None = None


class EmployeeVerificationRequest(BaseModel):
    """Request schema for employee verification."""

    employee_id: int = Field(..., description="Expected employee ID")


class EmployeeVerificationResponse(BaseModel):
    """Response schema for employee verification."""

    success: bool
    employee_id: int
    is_verified: bool
    confidence: float
    processing_time: float
    error: str | None = None


class FaceEmbeddingInfo(BaseModel):
    """Information about a face embedding."""

    point_id: str
    created_at: datetime | None = None
    metadata: dict = {}


class EmployeeFacesResponse(BaseModel):
    """Response schema for employee faces list."""

    employee_id: int
    face_count: int
    embeddings: list[FaceEmbeddingInfo] = []


class FaceIdentificationResponse(BaseModel):
    """Response schema for face identification."""

    success: bool
    employee_id: int | None
    confidence: float | None
    processing_time: float
    error: str | None = None
