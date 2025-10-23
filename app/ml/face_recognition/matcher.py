"""Face matching service integrating with Qdrant vector database."""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID
import numpy as np

from app.vector_db.face_vector_service import face_vector_service
from app.ml.face_recognition.extractor import (
    get_embedding_extractor,
    EmbeddingExtractor,
)
from app.ml.face_recognition.detector import get_face_detector, FaceDetector

logger = logging.getLogger(__name__)


class FaceMatchResult:
    """Container for face matching results."""

    def __init__(
        self,
        employee_id: Optional[UUID],
        confidence: float,
        is_match: bool,
        all_candidates: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize match result.

        Args:
            employee_id: Matched employee ID (None if no match)
            confidence: Confidence score (similarity score)
            is_match: Whether a match was found above threshold
            all_candidates: List of all candidate matches
        """
        self.employee_id = employee_id
        self.confidence = confidence
        self.is_match = is_match
        self.all_candidates = all_candidates or []

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "employee_id": str(self.employee_id) if self.employee_id else None,
            "confidence": float(self.confidence),
            "is_match": self.is_match,
            "candidates_count": len(self.all_candidates),
        }


class FaceMatcher:
    """Service for face matching using embeddings and Qdrant."""

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        detector: Optional[FaceDetector] = None,
        extractor: Optional[EmbeddingExtractor] = None,
    ):
        """
        Initialize face matcher.

        Args:
            similarity_threshold: Minimum similarity for a match (0.0-1.0)
            detector: Optional face detector instance
            extractor: Optional embedding extractor instance
        """
        self.similarity_threshold = similarity_threshold
        self.detector = detector or get_face_detector()
        self.extractor = extractor or get_embedding_extractor()

    async def register_face(
        self,
        employee_id: UUID,
        face_image: np.ndarray,
        check_duplicate: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a face embedding for an employee.

        Args:
            employee_id: UUID of the employee
            face_image: Face image in BGR format
            check_duplicate: Check if face already registered
            metadata: Additional metadata to store

        Returns:
            Dict with registration info: {
                "point_id": str,
                "employee_id": str,
                "is_duplicate": bool,
                "duplicate_employee_id": Optional[str]
            }

        Raises:
            ValueError: If no face detected or duplicate found
            RuntimeError: If registration fails
        """
        try:
            # Detect face
            detection = self.detector.detect_single_face(face_image, align_face=True)

            if detection is None:
                raise ValueError("No face detected in the image")

            # Extract embedding from aligned face
            if detection.aligned_face is None:
                raise ValueError("Face alignment failed")

            embedding = self.extractor.extract_embedding(
                detection.aligned_face, normalize=True
            )

            # Check for duplicates if requested
            if check_duplicate:
                match_result = await self.match_face_by_embedding(
                    embedding,
                    threshold=0.85,  # Higher threshold for duplicate detection
                )

                if match_result.is_match:
                    # Found potential duplicate
                    raise ValueError(
                        f"Face already registered for employee "
                        f"{match_result.employee_id} with similarity "
                        f"{match_result.confidence:.3f}"
                    )

            # Store embedding in Qdrant
            point_id = await face_vector_service.insert_embedding(
                employee_id=employee_id,
                embedding=embedding.tolist(),
                metadata=metadata or {},
            )

            logger.info(
                f"Registered face for employee {employee_id}, point_id: {point_id}"
            )

            return {
                "point_id": point_id,
                "employee_id": str(employee_id),
                "is_duplicate": False,
                "duplicate_employee_id": None,
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Face registration failed: {e}")
            raise RuntimeError(f"Face registration error: {e}") from e

    async def match_face(
        self,
        face_image: np.ndarray,
        threshold: Optional[float] = None,
        top_k: int = 5,
    ) -> FaceMatchResult:
        """
        Match a face image against registered faces.

        Args:
            face_image: Face image in BGR format
            threshold: Override similarity threshold
            top_k: Number of top candidates to retrieve

        Returns:
            FaceMatchResult with match information

        Raises:
            ValueError: If no face detected
            RuntimeError: If matching fails
        """
        try:
            # Detect face
            detection = self.detector.detect_single_face(face_image, align_face=True)

            if detection is None:
                logger.info("No face detected for matching")
                return FaceMatchResult(
                    employee_id=None,
                    confidence=0.0,
                    is_match=False,
                    all_candidates=[],
                )

            # Extract embedding
            if detection.aligned_face is None:
                raise ValueError("Face alignment failed")

            embedding = self.extractor.extract_embedding(
                detection.aligned_face, normalize=True
            )

            # Match against database
            return await self.match_face_by_embedding(
                embedding, threshold=threshold, top_k=top_k
            )

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Face matching failed: {e}")
            raise RuntimeError(f"Face matching error: {e}") from e

    async def match_face_by_embedding(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
        top_k: int = 5,
    ) -> FaceMatchResult:
        """
        Match a face embedding against registered faces.

        Args:
            embedding: Face embedding vector (512-dimensional)
            threshold: Override similarity threshold
            top_k: Number of top candidates to retrieve

        Returns:
            FaceMatchResult with match information
        """
        match_threshold = threshold or self.similarity_threshold

        try:
            # Search in Qdrant
            results = await face_vector_service.search_similar(
                query_embedding=embedding.tolist(),
                limit=top_k,
                score_threshold=match_threshold,
            )

            if not results:
                logger.info(f"No match found above threshold {match_threshold}")
                return FaceMatchResult(
                    employee_id=None,
                    confidence=0.0,
                    is_match=False,
                    all_candidates=[],
                )

            # Get top match
            top_match = results[0]
            employee_id = UUID(top_match["employee_id"])
            confidence = top_match["score"]

            logger.info(
                f"Face matched to employee {employee_id} with confidence {confidence:.3f}"
            )

            return FaceMatchResult(
                employee_id=employee_id,
                confidence=confidence,
                is_match=True,
                all_candidates=results,
            )

        except Exception as e:
            logger.error(f"Face matching by embedding failed: {e}")
            raise RuntimeError(f"Face matching error: {e}") from e

    async def match_multiple_faces(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None,
        max_faces: Optional[int] = None,
    ) -> List[FaceMatchResult]:
        """
        Detect and match multiple faces in an image.

        Args:
            image: Input image in BGR format
            threshold: Override similarity threshold
            max_faces: Maximum number of faces to process

        Returns:
            List of FaceMatchResult for each detected face
        """
        try:
            # Detect all faces
            detections = self.detector.detect_faces(
                image, align_faces=True, max_faces=max_faces
            )

            if not detections:
                logger.info("No faces detected in image")
                return []

            # Match each face
            results = []
            for detection in detections:
                if detection.aligned_face is None:
                    logger.warning("Skipping face without alignment")
                    continue

                embedding = self.extractor.extract_embedding(
                    detection.aligned_face, normalize=True
                )

                match_result = await self.match_face_by_embedding(
                    embedding, threshold=threshold
                )
                results.append(match_result)

            logger.info(
                f"Matched {sum(1 for r in results if r.is_match)}/{len(results)} faces"
            )

            return results

        except Exception as e:
            logger.error(f"Multiple face matching failed: {e}")
            raise RuntimeError(f"Multiple face matching error: {e}") from e

    async def delete_employee_faces(self, employee_id: UUID) -> int:
        """
        Delete all face embeddings for an employee.

        Args:
            employee_id: UUID of the employee

        Returns:
            Number of embeddings deleted
        """
        try:
            count = await face_vector_service.delete_embeddings_by_employee(
                employee_id=employee_id
            )
            logger.info(f"Deleted {count} face embeddings for employee {employee_id}")
            return count

        except Exception as e:
            logger.error(f"Failed to delete employee faces: {e}")
            raise RuntimeError(f"Delete employee faces error: {e}") from e

    async def get_employee_face_count(self, employee_id: UUID) -> int:
        """
        Get number of registered faces for an employee.

        Args:
            employee_id: UUID of the employee

        Returns:
            Number of registered faces
        """
        try:
            count = await face_vector_service.count_embeddings(employee_id=employee_id)
            return count

        except Exception as e:
            logger.error(f"Failed to count employee faces: {e}")
            raise RuntimeError(f"Count employee faces error: {e}") from e


# Global matcher instance
face_matcher = FaceMatcher(similarity_threshold=0.65)
