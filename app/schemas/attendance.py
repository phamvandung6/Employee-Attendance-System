"""Attendance schemas."""

from datetime import datetime, date
from pydantic import BaseModel, Field


class AttendanceBase(BaseModel):
    """Base attendance schema."""

    employee_id: int
    date: date
    status: str = "present"
    notes: str | None = None


class AttendanceCreate(AttendanceBase):
    """Attendance creation schema."""

    check_in: datetime | None = None
    check_out: datetime | None = None
    check_in_image_url: str | None = None
    check_out_image_url: str | None = None


class AttendanceUpdate(BaseModel):
    """Attendance update schema."""

    check_out: datetime | None = None
    check_out_image_url: str | None = None
    status: str | None = None
    notes: str | None = None


class AttendanceResponse(AttendanceBase):
    """Attendance response schema."""

    id: int
    check_in: datetime | None
    check_out: datetime | None
    check_in_image_url: str | None
    check_out_image_url: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AttendanceListResponse(BaseModel):
    """Paginated attendance list response."""

    items: list[AttendanceResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class CheckInRequest(BaseModel):
    """Check-in request schema."""

    employee_id: int | None = Field(
        None, description="Employee ID (if not using face recognition)"
    )


class CheckInResponse(BaseModel):
    """Check-in response schema."""

    success: bool
    attendance_id: int | None
    employee_id: int
    check_in_time: datetime
    message: str
    is_late: bool = False
    error: str | None = None


class CheckOutRequest(BaseModel):
    """Check-out request schema."""

    employee_id: int | None = Field(
        None, description="Employee ID (if not using face recognition)"
    )


class CheckOutResponse(BaseModel):
    """Check-out response schema."""

    success: bool
    attendance_id: int | None
    employee_id: int
    check_out_time: datetime
    work_duration: float | None = None  # in hours
    message: str
    error: str | None = None


class AttendanceSummary(BaseModel):
    """Attendance summary for an employee."""

    employee_id: int
    total_days: int
    present_days: int
    absent_days: int
    late_days: int
    attendance_rate: float  # percentage
