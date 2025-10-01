"""API v1 router."""

from fastapi import APIRouter

# from app.api.v1.endpoints import (
#     auth,
#     employees,
#     attendance,
#     face_recognition,
#     violations,
#     reports,
# )

api_router = APIRouter()

# Will be uncommented when endpoints are implemented
# api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
# api_router.include_router(employees.router, prefix="/employees", tags=["employees"])
# api_router.include_router(attendance.router, prefix="/attendance", tags=["attendance"])
# api_router.include_router(face_recognition.router, prefix="/face-recognition", tags=["face-recognition"])
# api_router.include_router(violations.router, prefix="/violations", tags=["violations"])
# api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
