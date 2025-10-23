"""Attendance repository for database operations."""

from datetime import date
from typing import Optional, Sequence
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attendance import Attendance


class AttendanceRepository:
    """Repository for Attendance model operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, attendance_id: int) -> Optional[Attendance]:
        """Get attendance by ID."""
        result = await self.db.execute(
            select(Attendance).where(Attendance.id == attendance_id)
        )
        return result.scalar_one_or_none()

    async def get_by_employee_and_date(
        self, employee_id: int, attendance_date: date
    ) -> Optional[Attendance]:
        """Get attendance record for specific employee and date."""
        result = await self.db.execute(
            select(Attendance).where(
                and_(
                    Attendance.employee_id == employee_id,
                    Attendance.date == attendance_date,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        employee_id: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        status: Optional[str] = None,
    ) -> tuple[Sequence[Attendance], int]:
        """
        Get all attendance records with pagination and filtering.

        Returns:
            Tuple of (attendances, total_count)
        """
        # Base query
        query = select(Attendance)
        count_query = select(func.count()).select_from(Attendance)

        # Apply filters
        filters = []

        if employee_id:
            filters.append(Attendance.employee_id == employee_id)

        if start_date:
            filters.append(Attendance.date >= start_date)

        if end_date:
            filters.append(Attendance.date <= end_date)

        if status:
            filters.append(Attendance.status == status)

        # Apply filters to both queries
        if filters:
            query = query.where(and_(*filters))
            count_query = count_query.where(and_(*filters))

        # Get total count
        total_result = await self.db.execute(count_query)
        total = total_result.scalar_one()

        # Apply pagination and ordering
        query = query.order_by(Attendance.date.desc()).offset(skip).limit(limit)

        # Execute query
        result = await self.db.execute(query)
        attendances = result.scalars().all()

        return attendances, total

    async def create(self, attendance: Attendance) -> Attendance:
        """Create a new attendance record."""
        self.db.add(attendance)
        await self.db.commit()
        await self.db.refresh(attendance)
        return attendance

    async def update(self, attendance: Attendance) -> Attendance:
        """Update attendance record."""
        await self.db.commit()
        await self.db.refresh(attendance)
        return attendance

    async def delete(self, attendance: Attendance) -> None:
        """Delete attendance record."""
        await self.db.delete(attendance)
        await self.db.commit()

    async def get_attendance_summary(
        self,
        employee_id: int,
        start_date: date,
        end_date: date,
    ) -> dict:
        """
        Get attendance summary for an employee in a date range.

        Returns:
            Dict with summary statistics
        """
        result = await self.db.execute(
            select(
                func.count(Attendance.id).label("total_days"),
                func.count(
                    Attendance.id if Attendance.status == "present" else None
                ).label("present_days"),
                func.count(
                    Attendance.id if Attendance.status == "late" else None
                ).label("late_days"),
            ).where(
                and_(
                    Attendance.employee_id == employee_id,
                    Attendance.date >= start_date,
                    Attendance.date <= end_date,
                )
            )
        )

        row = result.first()

        if row:
            total_days = row.total_days or 0
            present_days = row.present_days or 0
            late_days = row.late_days or 0

            return {
                "total_days": total_days,
                "present_days": present_days,
                "late_days": late_days,
                "absent_days": 0,  # Would need business day calculation
                "attendance_rate": (
                    (present_days / total_days * 100) if total_days > 0 else 0.0
                ),
            }

        return {
            "total_days": 0,
            "present_days": 0,
            "late_days": 0,
            "absent_days": 0,
            "attendance_rate": 0.0,
        }
