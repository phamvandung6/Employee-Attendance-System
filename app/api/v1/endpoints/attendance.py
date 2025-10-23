"""Attendance API endpoints."""

import logging
from datetime import date, datetime
from typing import Annotated
from fastapi import (
    APIRouter,
    Depends,
    Query,
    File,
    UploadFile,
    Form,
    HTTPException,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.models.user import User
from app.services.attendance_service import AttendanceService
from app.services.employee_service import EmployeeService
from app.services.storage_service import storage_service
from app.ml.face_recognition.pipeline import face_recognition_pipeline
from app.ml.face_recognition.preprocessor import preprocessor
from app.schemas.attendance import (
    AttendanceListResponse,
    AttendanceResponse,
    CheckInResponse,
    CheckOutResponse,
    AttendanceSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/attendance", tags=["attendance"])


@router.get("", response_model=AttendanceListResponse)
async def get_attendances(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    employee_id: int | None = Query(None, description="Filter by employee ID"),
    start_date: date | None = Query(None, description="Filter from date"),
    end_date: date | None = Query(None, description="Filter to date"),
    status: str | None = Query(
        None, description="Filter by status (present, absent, late, half_day)"
    ),
):
    """
    Get list of attendance records with pagination and filtering.

    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 50, max: 100)
    - **employee_id**: Filter by specific employee
    - **start_date**: Filter from date (YYYY-MM-DD)
    - **end_date**: Filter to date (YYYY-MM-DD)
    - **status**: Filter by status
    """
    service = AttendanceService(db)
    return await service.get_attendances(
        page=page,
        page_size=page_size,
        employee_id=employee_id,
        start_date=start_date,
        end_date=end_date,
        status=status,
    )


@router.get("/{attendance_id}", response_model=AttendanceResponse)
async def get_attendance(
    attendance_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get attendance record by ID."""
    service = AttendanceService(db)
    attendance = await service.get_attendance(attendance_id)
    return AttendanceResponse.model_validate(attendance)


@router.post("/check-in", response_model=CheckInResponse)
async def check_in(
    image: Annotated[UploadFile, File(..., description="Face image for recognition")],
    employee_id: Annotated[
        int | None, Form(description="Employee ID (optional if using face recognition)")
    ] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Check-in employee using face recognition or manual ID.

    - **image**: Face image for recognition
    - **employee_id**: Optional employee ID (if not using face recognition)

    Automatically detects late check-ins and prevents duplicate check-ins.
    """
    try:
        service = AttendanceService(db)
        employee_service = EmployeeService(db)

        # Validate image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload an image.",
            )

        # Read and decode image
        image_bytes = await image.read()
        img = preprocessor.decode_image(image_bytes)

        # If no employee_id provided, use face recognition
        if not employee_id:
            result = await face_recognition_pipeline.identify_employee(img)

            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No matching employee found. Please try again or use manual check-in.",
                )

            identified_id, confidence = result
            employee_id = int(identified_id.int)

            logger.info(
                f"Identified employee {employee_id} with confidence {confidence:.3f}"
            )

        # Verify employee exists
        employee = await employee_service.get_employee(employee_id)

        # Upload image to storage
        image_url = None
        try:
            await image.seek(0)
            upload_result = await storage_service.upload_file(
                file=image,
                folder="attendance/check-in",
                filename_prefix=f"emp_{employee_id}",
                validate_image=True,
            )
            image_url = upload_result["url"]
        except Exception as e:
            logger.warning(f"Failed to store check-in image: {e}")

        # Process check-in
        attendance, is_late = await service.check_in(
            employee_id=employee_id,
            check_in_time=datetime.now(),
            image_url=image_url,
        )

        return CheckInResponse(
            success=True,
            attendance_id=attendance.id,
            employee_id=employee_id,
            check_in_time=attendance.check_in,
            message=f"Check-in successful for {employee.name}"
            + (" (Late)" if is_late else ""),
            is_late=is_late,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Check-in validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Check-in failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during check-in: {str(e)}",
        )


@router.post("/check-out", response_model=CheckOutResponse)
async def check_out(
    image: Annotated[UploadFile, File(..., description="Face image for recognition")],
    employee_id: Annotated[
        int | None, Form(description="Employee ID (optional if using face recognition)")
    ] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Check-out employee using face recognition or manual ID.

    - **image**: Face image for recognition
    - **employee_id**: Optional employee ID (if not using face recognition)

    Calculates work duration and prevents duplicate check-outs.
    """
    try:
        service = AttendanceService(db)
        employee_service = EmployeeService(db)

        # Validate image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload an image.",
            )

        # Read and decode image
        image_bytes = await image.read()
        img = preprocessor.decode_image(image_bytes)

        # If no employee_id provided, use face recognition
        if not employee_id:
            result = await face_recognition_pipeline.identify_employee(img)

            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No matching employee found. Please try again or use manual check-out.",
                )

            identified_id, confidence = result
            employee_id = int(identified_id.int)

            logger.info(
                f"Identified employee {employee_id} with confidence {confidence:.3f}"
            )

        # Verify employee exists
        employee = await employee_service.get_employee(employee_id)

        # Upload image to storage
        image_url = None
        try:
            await image.seek(0)
            upload_result = await storage_service.upload_file(
                file=image,
                folder="attendance/check-out",
                filename_prefix=f"emp_{employee_id}",
                validate_image=True,
            )
            image_url = upload_result["url"]
        except Exception as e:
            logger.warning(f"Failed to store check-out image: {e}")

        # Process check-out
        attendance, work_duration = await service.check_out(
            employee_id=employee_id,
            check_out_time=datetime.now(),
            image_url=image_url,
        )

        return CheckOutResponse(
            success=True,
            attendance_id=attendance.id,
            employee_id=employee_id,
            check_out_time=attendance.check_out,
            work_duration=round(work_duration, 2),
            message=f"Check-out successful for {employee.name}. "
            f"Work duration: {work_duration:.2f} hours",
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Check-out validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Check-out failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during check-out: {str(e)}",
        )


@router.get("/summary/{employee_id}", response_model=AttendanceSummary)
async def get_attendance_summary(
    employee_id: int,
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get attendance summary for an employee in a date range.

    - **employee_id**: Employee ID
    - **start_date**: Start date (inclusive)
    - **end_date**: End date (inclusive)

    Returns attendance statistics including rate, present/absent/late days.
    """
    try:
        employee_service = EmployeeService(db)
        await employee_service.get_employee(employee_id)

        service = AttendanceService(db)
        return await service.get_attendance_summary(employee_id, start_date, end_date)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get attendance summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )
