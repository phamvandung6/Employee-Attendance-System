"""Attendance model."""

from datetime import datetime, date
from sqlalchemy import Date, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel


class Attendance(BaseModel):
    """Attendance record model."""

    __tablename__ = "attendances"

    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    check_in: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    check_out: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    check_in_image_url: Mapped[str] = mapped_column(String(512), nullable=True)
    check_out_image_url: Mapped[str] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="present", nullable=False
    )  # present, absent, late, half_day
    notes: Mapped[str] = mapped_column(String(500), nullable=True)

    # Relationships
    employee: Mapped["Employee"] = relationship(
        "Employee", back_populates="attendances"
    )

    def __repr__(self) -> str:
        return f"<Attendance(id={self.id}, employee_id={self.employee_id}, date={self.date})>"
