"""Violation model."""

from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel


class Violation(BaseModel):
    """Violation record model."""

    __tablename__ = "violations"

    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    violation_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # no_id_card, no_uniform, both
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    image_url: Mapped[str] = mapped_column(String(512), nullable=True)
    severity: Mapped[str] = mapped_column(
        String(20), default="low", nullable=False
    )  # low, medium, high
    description: Mapped[str] = mapped_column(Text, nullable=True)
    resolved: Mapped[bool] = mapped_column(default=False, nullable=False)
    resolved_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    resolved_by: Mapped[int] = mapped_column(nullable=True)  # User ID who resolved

    # Relationships
    employee: Mapped["Employee"] = relationship("Employee", back_populates="violations")

    def __repr__(self) -> str:
        return f"<Violation(id={self.id}, type={self.violation_type}, employee_id={self.employee_id})>"
