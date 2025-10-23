"""API v1 router."""

from fastapi import APIRouter
from app.api.v1.endpoints import auth, employees, face_recognition, attendance

api_router = APIRouter()

# Authentication routes
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])

# Employee management routes
api_router.include_router(employees.router, prefix="/employees", tags=["employees"])

# Face recognition routes
api_router.include_router(
    face_recognition.router, prefix="/face-recognition", tags=["face-recognition"]
)

# Attendance routes
api_router.include_router(attendance.router, prefix="/attendance", tags=["attendance"])
# api_router.include_router(violations.router, prefix="/violations", tags=["violations"])
# api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
