"""Employee repository for database operations."""

from typing import Optional, Sequence
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.employee import Employee


class EmployeeRepository:
    """Repository for Employee model operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, employee_id: int) -> Optional[Employee]:
        """Get employee by ID."""
        result = await self.db.execute(
            select(Employee).where(Employee.id == employee_id)
        )
        return result.scalar_one_or_none()

    async def get_by_employee_code(self, employee_code: str) -> Optional[Employee]:
        """Get employee by employee code."""
        result = await self.db.execute(
            select(Employee).where(Employee.employee_code == employee_code)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[Employee]:
        """Get employee by email."""
        result = await self.db.execute(select(Employee).where(Employee.email == email))
        return result.scalar_one_or_none()

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
        department: Optional[str] = None,
        active_only: bool = False,
    ) -> tuple[Sequence[Employee], int]:
        """
        Get all employees with pagination and filtering.

        Returns:
            Tuple of (employees, total_count)
        """
        # Base query
        query = select(Employee)
        count_query = select(func.count()).select_from(Employee)

        # Apply filters
        filters = []
        if search:
            search_filter = or_(
                Employee.name.ilike(f"%{search}%"),
                Employee.employee_code.ilike(f"%{search}%"),
                Employee.email.ilike(f"%{search}%") if search else False,
            )
            filters.append(search_filter)

        if department:
            filters.append(Employee.department == department)

        if active_only:
            filters.append(Employee.active_status == True)

        # Apply filters to both queries
        if filters:
            query = query.where(*filters)
            count_query = count_query.where(*filters)

        # Get total count
        total_result = await self.db.execute(count_query)
        total = total_result.scalar_one()

        # Apply pagination and ordering
        query = query.order_by(Employee.created_at.desc()).offset(skip).limit(limit)

        # Execute query
        result = await self.db.execute(query)
        employees = result.scalars().all()

        return employees, total

    async def create(self, employee: Employee) -> Employee:
        """Create a new employee."""
        self.db.add(employee)
        await self.db.commit()
        await self.db.refresh(employee)
        return employee

    async def update(self, employee: Employee) -> Employee:
        """Update employee."""
        await self.db.commit()
        await self.db.refresh(employee)
        return employee

    async def delete(self, employee: Employee) -> None:
        """Delete employee."""
        await self.db.delete(employee)
        await self.db.commit()

    async def get_departments(self) -> list[str]:
        """Get list of unique departments."""
        result = await self.db.execute(
            select(Employee.department)
            .distinct()
            .where(Employee.department.is_not(None))
            .order_by(Employee.department)
        )
        return list(result.scalars().all())
