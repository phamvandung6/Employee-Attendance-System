"""Employee service with business logic."""

import math
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError
from app.models.employee import Employee
from app.repositories.employee_repository import EmployeeRepository
from app.schemas.employee import (
    EmployeeCreate,
    EmployeeUpdate,
    EmployeeListResponse,
    EmployeeResponse,
)


class EmployeeService:
    """Service for employee operations."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.employee_repo = EmployeeRepository(db)

    async def get_employee(self, employee_id: int) -> Employee:
        """Get employee by ID."""
        employee = await self.employee_repo.get_by_id(employee_id)
        if not employee:
            raise AuthenticationError(f"Employee with ID {employee_id} not found")
        return employee

    async def get_employees(
        self,
        page: int = 1,
        page_size: int = 50,
        search: str | None = None,
        department: str | None = None,
        active_only: bool = False,
    ) -> EmployeeListResponse:
        """
        Get paginated list of employees with filtering.

        Args:
            page: Page number (starts from 1)
            page_size: Number of items per page
            search: Search term for name, code, or email
            department: Filter by department
            active_only: Only return active employees

        Returns:
            EmployeeListResponse with pagination metadata
        """
        # Calculate offset
        skip = (page - 1) * page_size

        # Get employees and total count
        employees, total = await self.employee_repo.get_all(
            skip=skip,
            limit=page_size,
            search=search,
            department=department,
            active_only=active_only,
        )

        # Calculate total pages
        total_pages = math.ceil(total / page_size) if total > 0 else 0

        return EmployeeListResponse(
            items=[EmployeeResponse.model_validate(emp) for emp in employees],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    async def create_employee(self, employee_data: EmployeeCreate) -> Employee:
        """Create a new employee with validation."""
        # Check if employee code already exists
        existing_code = await self.employee_repo.get_by_employee_code(
            employee_data.employee_code
        )
        if existing_code:
            raise AuthenticationError(
                f"Employee code '{employee_data.employee_code}' already exists"
            )

        # Check if email already exists (if provided)
        if employee_data.email:
            existing_email = await self.employee_repo.get_by_email(employee_data.email)
            if existing_email:
                raise AuthenticationError(
                    f"Email '{employee_data.email}' already registered"
                )

        # Create employee
        employee = Employee(
            name=employee_data.name,
            employee_code=employee_data.employee_code,
            department=employee_data.department,
            position=employee_data.position,
            email=employee_data.email,
            phone=employee_data.phone,
            active_status=employee_data.active_status,
        )

        return await self.employee_repo.create(employee)

    async def update_employee(
        self, employee_id: int, employee_data: EmployeeUpdate
    ) -> Employee:
        """Update employee with validation."""
        # Get existing employee
        employee = await self.get_employee(employee_id)

        # Check email uniqueness if being updated
        if employee_data.email and employee_data.email != employee.email:
            existing_email = await self.employee_repo.get_by_email(employee_data.email)
            if existing_email and existing_email.id != employee_id:
                raise AuthenticationError(
                    f"Email '{employee_data.email}' already registered"
                )

        # Update fields
        update_data = employee_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(employee, field, value)

        return await self.employee_repo.update(employee)

    async def delete_employee(self, employee_id: int) -> None:
        """Delete employee."""
        employee = await self.get_employee(employee_id)
        await self.employee_repo.delete(employee)

    async def get_departments(self) -> list[str]:
        """Get list of all unique departments."""
        return await self.employee_repo.get_departments()
