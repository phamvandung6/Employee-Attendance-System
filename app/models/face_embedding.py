"""Face embedding model."""

from sqlalchemy import ForeignKey, String, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel


class FaceEmbedding(BaseModel):
    """Face embedding model for storing face vectors."""

    __tablename__ = "face_embeddings"

    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    embedding_vector: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False
    )  # Store as binary, will convert from numpy array
    image_url: Mapped[str] = mapped_column(String(512), nullable=False)
    qdrant_id: Mapped[str] = mapped_column(
        String(100), nullable=True, unique=True, index=True
    )
    quality_score: Mapped[float] = mapped_column(
        nullable=True
    )  # Face image quality score
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    # Relationships
    employee: Mapped["Employee"] = relationship(
        "Employee", back_populates="face_embeddings"
    )

    def __repr__(self) -> str:
        return f"<FaceEmbedding(id={self.id}, employee_id={self.employee_id}, qdrant_id={self.qdrant_id})>"
