"""Employee model."""

from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List

from app.models.base import BaseModel


class Employee(BaseModel):
    """Employee model."""

    __tablename__ = "employees"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    employee_code: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    department: Mapped[str] = mapped_column(String(100), nullable=True)
    position: Mapped[str] = mapped_column(String(100), nullable=True)
    email: Mapped[str] = mapped_column(
        String(255), nullable=True, unique=True, index=True
    )
    phone: Mapped[str] = mapped_column(String(20), nullable=True)
    active_status: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    attendances: Mapped[List["Attendance"]] = relationship(
        "Attendance", back_populates="employee", cascade="all, delete-orphan"
    )
    face_embeddings: Mapped[List["FaceEmbedding"]] = relationship(
        "FaceEmbedding", back_populates="employee", cascade="all, delete-orphan"
    )
    violations: Mapped[List["Violation"]] = relationship(
        "Violation", back_populates="employee", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Employee(id={self.id}, name={self.name}, code={self.employee_code})>"
