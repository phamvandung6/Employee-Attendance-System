"""Face recognition API endpoints."""

import logging
from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.models.user import User
from app.services.employee_service import EmployeeService
from app.services.storage_service import storage_service
from app.ml.face_recognition.pipeline import face_recognition_pipeline
from app.ml.face_recognition.preprocessor import preprocessor
from app.schemas.face_recognition import (
    FaceRegistrationResponse,
    FaceRecognitionResponse,
    FaceMatchInfo,
    EmployeeVerificationResponse,
    EmployeeFacesResponse,
    FaceIdentificationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/register", response_model=FaceRegistrationResponse)
async def register_face(
    employee_id: Annotated[int, Form(..., description="Employee ID")],
    image: Annotated[UploadFile, File(..., description="Face image file")],
    check_duplicate: Annotated[
        bool, Form(description="Check for duplicate faces")
    ] = True,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Register a face for an employee.

    - **employee_id**: ID of the employee
    - **image**: Face image file (JPG, PNG, WEBP)
    - **check_duplicate**: Whether to check for duplicate registrations

    Returns registration result with point_id and processing time.
    """
    try:
        # Verify employee exists
        employee_service = EmployeeService(db)
        employee = await employee_service.get_employee(employee_id)

        # Validate and read image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload an image.",
            )

        # Read image bytes
        image_bytes = await image.read()

        # Decode image
        img = preprocessor.decode_image(image_bytes)

        # Register face through pipeline
        result = await face_recognition_pipeline.register_employee_face(
            employee_id=UUID(int=employee_id),
            image=img,
            check_duplicate=check_duplicate,
            metadata={"employee_code": employee.employee_code, "name": employee.name},
        )

        # Store image if registration successful
        if result.success:
            try:
                # Reset file pointer
                await image.seek(0)

                # Upload to storage
                upload_result = await storage_service.upload_file(
                    file=image,
                    folder="faces",
                    filename_prefix=f"emp_{employee_id}",
                    validate_image=True,
                )

                logger.info(
                    f"Stored face image for employee {employee_id}: "
                    f"{upload_result['object_name']}"
                )

            except Exception as e:
                logger.warning(f"Failed to store face image: {e}")
                # Continue even if storage fails

        return FaceRegistrationResponse(
            success=result.success,
            employee_id=employee_id,
            point_id=result.point_id,
            processing_time=result.processing_time,
            message="Face registered successfully"
            if result.success
            else "Registration failed",
            image_quality=result.image_quality,
            error=result.error,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Face registration validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Face registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during registration: {str(e)}",
        )


@router.post("/recognize", response_model=FaceRecognitionResponse)
async def recognize_faces(
    image: Annotated[UploadFile, File(..., description="Image containing faces")],
    max_faces: Annotated[
        int | None, Form(description="Maximum faces to detect")
    ] = None,
    threshold: Annotated[
        float | None, Form(ge=0.0, le=1.0, description="Match threshold")
    ] = None,
    current_user: User = Depends(get_current_user),
):
    """
    Recognize faces in an image.

    - **image**: Image file containing one or more faces
    - **max_faces**: Maximum number of faces to process (optional)
    - **threshold**: Similarity threshold for matching (0.0-1.0, optional)

    Returns list of recognized employees with confidence scores.
    """
    try:
        # Validate image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload an image.",
            )

        # Read and decode image
        image_bytes = await image.read()
        img = preprocessor.decode_image(image_bytes)

        # Process through pipeline
        result = await face_recognition_pipeline.process_image(
            image=img, max_faces=max_faces, threshold=threshold
        )

        # Convert matches to response format
        matches = [
            FaceMatchInfo(
                employee_id=int(match.employee_id.int) if match.employee_id else None,
                confidence=match.confidence,
                is_match=match.is_match,
            )
            for match in result.matches
        ]

        return FaceRecognitionResponse(
            success=result.success,
            processing_time=result.processing_time,
            faces_detected=result.faces_detected,
            faces_matched=result.faces_matched,
            matches=matches,
            error=result.error,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Face recognition validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Face recognition failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during recognition: {str(e)}",
        )


@router.post("/verify/{employee_id}", response_model=EmployeeVerificationResponse)
async def verify_employee(
    employee_id: int,
    image: Annotated[UploadFile, File(..., description="Face image to verify")],
    threshold: Annotated[
        float | None, Form(ge=0.0, le=1.0, description="Match threshold")
    ] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Verify if an image matches a specific employee.

    - **employee_id**: Expected employee ID
    - **image**: Face image to verify
    - **threshold**: Similarity threshold (optional)

    Returns verification result with confidence score.
    """
    try:
        # Verify employee exists
        employee_service = EmployeeService(db)
        await employee_service.get_employee(employee_id)

        # Validate and read image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload an image.",
            )

        image_bytes = await image.read()
        img = preprocessor.decode_image(image_bytes)

        # Verify through pipeline
        import time

        start_time = time.time()
        is_verified, confidence = await face_recognition_pipeline.verify_employee(
            employee_id=UUID(int=employee_id), image=img, threshold=threshold
        )
        processing_time = time.time() - start_time

        return EmployeeVerificationResponse(
            success=True,
            employee_id=employee_id,
            is_verified=is_verified,
            confidence=confidence,
            processing_time=processing_time,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Employee verification validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Employee verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during verification: {str(e)}",
        )


@router.post("/identify", response_model=FaceIdentificationResponse)
async def identify_employee(
    image: Annotated[UploadFile, File(..., description="Face image")],
    threshold: Annotated[
        float | None, Form(ge=0.0, le=1.0, description="Match threshold")
    ] = None,
    current_user: User = Depends(get_current_user),
):
    """
    Identify employee from a face image (single face expected).

    - **image**: Face image containing one face
    - **threshold**: Similarity threshold (optional)

    Returns identified employee ID and confidence.
    """
    try:
        # Validate image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload an image.",
            )

        image_bytes = await image.read()
        img = preprocessor.decode_image(image_bytes)

        # Identify through pipeline
        import time

        start_time = time.time()
        result = await face_recognition_pipeline.identify_employee(
            image=img, threshold=threshold
        )
        processing_time = time.time() - start_time

        if result:
            employee_id, confidence = result
            return FaceIdentificationResponse(
                success=True,
                employee_id=int(employee_id.int),
                confidence=confidence,
                processing_time=processing_time,
            )
        else:
            return FaceIdentificationResponse(
                success=False,
                employee_id=None,
                confidence=None,
                processing_time=processing_time,
                error="No matching employee found",
            )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Employee identification validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Employee identification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during identification: {str(e)}",
        )


@router.get("/embeddings/{employee_id}", response_model=EmployeeFacesResponse)
async def get_employee_faces(
    employee_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get face embeddings count for an employee.

    - **employee_id**: Employee ID

    Returns count of registered face embeddings.
    """
    try:
        # Verify employee exists
        employee_service = EmployeeService(db)
        await employee_service.get_employee(employee_id)

        # Get face count from matcher
        from app.ml.face_recognition.matcher import face_matcher

        count = await face_matcher.get_employee_face_count(UUID(int=employee_id))

        return EmployeeFacesResponse(
            employee_id=employee_id, face_count=count, embeddings=[]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get employee faces: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.delete("/embeddings/{employee_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_employee_faces(
    employee_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete all face embeddings for an employee.

    - **employee_id**: Employee ID

    This will remove all registered faces for the employee from the system.
    """
    try:
        # Verify employee exists
        employee_service = EmployeeService(db)
        await employee_service.get_employee(employee_id)

        # Delete faces through matcher
        from app.ml.face_recognition.matcher import face_matcher

        count = await face_matcher.delete_employee_faces(UUID(int=employee_id))

        logger.info(f"Deleted {count} face embeddings for employee {employee_id}")

        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete employee faces: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )
