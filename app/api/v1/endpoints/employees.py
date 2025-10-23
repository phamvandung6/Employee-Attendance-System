"""Employee management endpoints."""

from typing import Annotated
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.models.user import User
from app.services.employee_service import EmployeeService
from app.schemas.employee import (
    EmployeeCreate,
    EmployeeUpdate,
    EmployeeResponse,
    EmployeeListResponse,
)

router = APIRouter(prefix="/employees", tags=["employees"])


@router.get("", response_model=EmployeeListResponse)
async def get_employees(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    search: str | None = Query(None, description="Search by name, code, or email"),
    department: str | None = Query(None, description="Filter by department"),
    active_only: bool = Query(False, description="Show only active employees"),
):
    """
    Get list of employees with pagination and filtering.

    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 50, max: 100)
    - **search**: Search term for name, employee code, or email
    - **department**: Filter by specific department
    - **active_only**: Only return active employees
    """
    service = EmployeeService(db)
    return await service.get_employees(
        page=page,
        page_size=page_size,
        search=search,
        department=department,
        active_only=active_only,
    )


@router.get("/departments", response_model=list[str])
async def get_departments(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get list of all unique departments."""
    service = EmployeeService(db)
    return await service.get_departments()


@router.get("/{employee_id}", response_model=EmployeeResponse)
async def get_employee(
    employee_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get employee by ID."""
    service = EmployeeService(db)
    employee = await service.get_employee(employee_id)
    return EmployeeResponse.model_validate(employee)


@router.post("", response_model=EmployeeResponse, status_code=status.HTTP_201_CREATED)
async def create_employee(
    employee_data: EmployeeCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Create a new employee.

    Requires:
    - Unique employee_code
    - Unique email (if provided)
    """
    service = EmployeeService(db)
    employee = await service.create_employee(employee_data)
    return EmployeeResponse.model_validate(employee)


@router.put("/{employee_id}", response_model=EmployeeResponse)
async def update_employee(
    employee_id: int,
    employee_data: EmployeeUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Update employee information.

    All fields are optional. Only provided fields will be updated.
    """
    service = EmployeeService(db)
    employee = await service.update_employee(employee_id, employee_data)
    return EmployeeResponse.model_validate(employee)


@router.delete("/{employee_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_employee(
    employee_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Delete an employee.

    This will also cascade delete:
    - Face embeddings
    - Attendance records
    - Violation records
    """
    service = EmployeeService(db)
    await service.delete_employee(employee_id)
    return None
