"""Attendance service with business logic."""

import math
from datetime import datetime, date, time
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError
from app.models.attendance import Attendance
from app.repositories.attendance_repository import AttendanceRepository
from app.schemas.attendance import (
    AttendanceListResponse,
    AttendanceResponse,
    AttendanceSummary,
)


class AttendanceService:
    """Service for attendance operations."""

    # Business rules constants
    WORK_START_TIME = time(9, 0)  # 9:00 AM
    LATE_THRESHOLD_MINUTES = 15  # Late if check-in after 9:15 AM
    EARLY_CHECKOUT_THRESHOLD = time(17, 0)  # 5:00 PM

    def __init__(self, db: AsyncSession):
        self.db = db
        self.attendance_repo = AttendanceRepository(db)

    async def get_attendance(self, attendance_id: int) -> Attendance:
        """Get attendance by ID."""
        attendance = await self.attendance_repo.get_by_id(attendance_id)
        if not attendance:
            raise AuthenticationError(
                f"Attendance record with ID {attendance_id} not found"
            )
        return attendance

    async def get_attendances(
        self,
        page: int = 1,
        page_size: int = 50,
        employee_id: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        status: str | None = None,
    ) -> AttendanceListResponse:
        """
        Get paginated list of attendance records with filtering.

        Args:
            page: Page number (starts from 1)
            page_size: Number of items per page
            employee_id: Filter by employee ID
            start_date: Filter from date (inclusive)
            end_date: Filter to date (inclusive)
            status: Filter by status

        Returns:
            AttendanceListResponse with pagination metadata
        """
        # Calculate offset
        skip = (page - 1) * page_size

        # Get attendances and total count
        attendances, total = await self.attendance_repo.get_all(
            skip=skip,
            limit=page_size,
            employee_id=employee_id,
            start_date=start_date,
            end_date=end_date,
            status=status,
        )

        # Calculate total pages
        total_pages = math.ceil(total / page_size) if total > 0 else 0

        return AttendanceListResponse(
            items=[AttendanceResponse.model_validate(att) for att in attendances],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    async def check_in(
        self,
        employee_id: int,
        check_in_time: datetime | None = None,
        image_url: str | None = None,
    ) -> tuple[Attendance, bool]:
        """
        Process employee check-in.

        Args:
            employee_id: Employee ID
            check_in_time: Check-in time (defaults to now)
            image_url: URL of check-in image

        Returns:
            Tuple of (attendance_record, is_late)

        Raises:
            ValueError: If already checked in today
        """
        check_in_time = check_in_time or datetime.now()
        today = check_in_time.date()

        # Check if already checked in today
        existing = await self.attendance_repo.get_by_employee_and_date(
            employee_id, today
        )

        if existing and existing.check_in:
            raise ValueError(
                f"Employee already checked in today at "
                f"{existing.check_in.strftime('%H:%M:%S')}"
            )

        # Determine if late
        late_cutoff = datetime.combine(
            today,
            time(
                self.WORK_START_TIME.hour,
                self.WORK_START_TIME.minute + self.LATE_THRESHOLD_MINUTES,
            ),
        )
        is_late = check_in_time > late_cutoff

        # Create or update attendance
        if existing:
            # Update existing record
            existing.check_in = check_in_time
            existing.check_in_image_url = image_url
            existing.status = "late" if is_late else "present"
            attendance = await self.attendance_repo.update(existing)
        else:
            # Create new record
            attendance = Attendance(
                employee_id=employee_id,
                date=today,
                check_in=check_in_time,
                check_in_image_url=image_url,
                status="late" if is_late else "present",
            )
            attendance = await self.attendance_repo.create(attendance)

        return attendance, is_late

    async def check_out(
        self,
        employee_id: int,
        check_out_time: datetime | None = None,
        image_url: str | None = None,
    ) -> tuple[Attendance, float]:
        """
        Process employee check-out.

        Args:
            employee_id: Employee ID
            check_out_time: Check-out time (defaults to now)
            image_url: URL of check-out image

        Returns:
            Tuple of (attendance_record, work_duration_hours)

        Raises:
            ValueError: If not checked in today or already checked out
        """
        check_out_time = check_out_time or datetime.now()
        today = check_out_time.date()

        # Get today's attendance
        attendance = await self.attendance_repo.get_by_employee_and_date(
            employee_id, today
        )

        if not attendance or not attendance.check_in:
            raise ValueError("Employee has not checked in today")

        if attendance.check_out:
            raise ValueError(
                f"Employee already checked out today at "
                f"{attendance.check_out.strftime('%H:%M:%S')}"
            )

        # Calculate work duration
        work_duration = (check_out_time - attendance.check_in).total_seconds() / 3600

        # Update attendance
        attendance.check_out = check_out_time
        attendance.check_out_image_url = image_url

        # Update status if needed (e.g., half day if checked out too early)
        early_checkout = check_out_time.time() < self.EARLY_CHECKOUT_THRESHOLD
        if work_duration < 4 and early_checkout:
            attendance.status = "half_day"

        attendance = await self.attendance_repo.update(attendance)

        return attendance, work_duration

    async def get_today_attendance(self, employee_id: int) -> Attendance | None:
        """Get today's attendance for an employee."""
        today = date.today()
        return await self.attendance_repo.get_by_employee_and_date(employee_id, today)

    async def get_attendance_summary(
        self,
        employee_id: int,
        start_date: date,
        end_date: date,
    ) -> AttendanceSummary:
        """
        Get attendance summary for an employee.

        Args:
            employee_id: Employee ID
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            AttendanceSummary with statistics
        """
        summary = await self.attendance_repo.get_attendance_summary(
            employee_id, start_date, end_date
        )

        return AttendanceSummary(
            employee_id=employee_id,
            total_days=summary["total_days"],
            present_days=summary["present_days"],
            absent_days=summary["absent_days"],
            late_days=summary["late_days"],
            attendance_rate=summary["attendance_rate"],
        )

    async def mark_absent(
        self,
        employee_id: int,
        absence_date: date,
        notes: str | None = None,
    ) -> Attendance:
        """
        Mark employee as absent for a specific date.

        Args:
            employee_id: Employee ID
            absence_date: Date to mark absent
            notes: Optional notes

        Returns:
            Attendance record
        """
        # Check if attendance already exists
        existing = await self.attendance_repo.get_by_employee_and_date(
            employee_id, absence_date
        )

        if existing:
            # Update to absent
            existing.status = "absent"
            if notes:
                existing.notes = notes
            return await self.attendance_repo.update(existing)
        else:
            # Create new absent record
            attendance = Attendance(
                employee_id=employee_id,
                date=absence_date,
                status="absent",
                notes=notes,
            )
            return await self.attendance_repo.create(attendance)

    async def update_attendance(
        self,
        attendance_id: int,
        status: str | None = None,
        notes: str | None = None,
    ) -> Attendance:
        """
        Update attendance record.

        Args:
            attendance_id: Attendance ID
            status: New status
            notes: New notes

        Returns:
            Updated attendance
        """
        attendance = await self.get_attendance(attendance_id)

        if status:
            attendance.status = status
        if notes:
            attendance.notes = notes

        return await self.attendance_repo.update(attendance)
