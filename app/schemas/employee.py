"""Employee schemas."""

from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class EmployeeBase(BaseModel):
    """Base employee schema."""

    name: str = Field(..., min_length=1, max_length=255)
    employee_code: str = Field(..., min_length=1, max_length=50)
    department: str | None = Field(None, max_length=100)
    position: str | None = Field(None, max_length=100)
    email: EmailStr | None = None
    phone: str | None = Field(None, max_length=20)


class EmployeeCreate(EmployeeBase):
    """Employee creation schema."""

    active_status: bool = True


class EmployeeUpdate(BaseModel):
    """Employee update schema - all fields optional."""

    name: str | None = Field(None, min_length=1, max_length=255)
    department: str | None = Field(None, max_length=100)
    position: str | None = Field(None, max_length=100)
    email: EmailStr | None = None
    phone: str | None = Field(None, max_length=20)
    active_status: bool | None = None


class EmployeeInDB(EmployeeBase):
    """Employee schema with database fields."""

    id: int
    active_status: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EmployeeResponse(EmployeeBase):
    """Employee response schema."""

    id: int
    active_status: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EmployeeListResponse(BaseModel):
    """Paginated employee list response."""

    items: list[EmployeeResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

    class Config:
        from_attributes = True
