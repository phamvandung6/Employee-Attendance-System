"""Complete face recognition pipeline integrating all components."""

import logging
import time
from typing import List, Optional, Dict, Any
from uuid import UUID
import numpy as np

from app.ml.face_recognition.detector import get_face_detector, FaceDetector
from app.ml.face_recognition.extractor import (
    get_embedding_extractor,
    EmbeddingExtractor,
)
from app.ml.face_recognition.matcher import face_matcher, FaceMatcher, FaceMatchResult
from app.ml.face_recognition.preprocessor import preprocessor
from app.core.config import settings

logger = logging.getLogger(__name__)


class RecognitionResult:
    """Container for complete face recognition results."""

    def __init__(
        self,
        success: bool,
        processing_time: float,
        faces_detected: int,
        faces_matched: int,
        matches: List[FaceMatchResult],
        error: Optional[str] = None,
    ):
        """
        Initialize recognition result.

        Args:
            success: Whether processing was successful
            processing_time: Total processing time in seconds
            faces_detected: Number of faces detected
            faces_matched: Number of faces successfully matched
            matches: List of match results for each face
            error: Error message if failed
        """
        self.success = success
        self.processing_time = processing_time
        self.faces_detected = faces_detected
        self.faces_matched = faces_matched
        self.matches = matches
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "success": self.success,
            "processing_time": round(self.processing_time, 3),
            "faces_detected": self.faces_detected,
            "faces_matched": self.faces_matched,
            "matches": [match.to_dict() for match in self.matches],
            "error": self.error,
        }


class RegistrationResult:
    """Container for face registration results."""

    def __init__(
        self,
        success: bool,
        employee_id: Optional[UUID],
        point_id: Optional[str],
        processing_time: float,
        image_quality: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """
        Initialize registration result.

        Args:
            success: Whether registration was successful
            employee_id: Employee ID
            point_id: Qdrant point ID
            processing_time: Processing time in seconds
            image_quality: Optional image quality metrics
            error: Error message if failed
        """
        self.success = success
        self.employee_id = employee_id
        self.point_id = point_id
        self.processing_time = processing_time
        self.image_quality = image_quality
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "success": self.success,
            "employee_id": str(self.employee_id) if self.employee_id else None,
            "point_id": self.point_id,
            "processing_time": round(self.processing_time, 3),
            "image_quality": self.image_quality,
            "error": self.error,
        }


class FaceRecognitionPipeline:
    """Complete end-to-end face recognition pipeline."""

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        extractor: Optional[EmbeddingExtractor] = None,
        matcher: Optional[FaceMatcher] = None,
        match_threshold: float = 0.65,
    ):
        """
        Initialize face recognition pipeline.

        Args:
            detector: Optional face detector instance
            extractor: Optional embedding extractor instance
            matcher: Optional face matcher instance
            match_threshold: Similarity threshold for matching
        """
        self.detector = detector or get_face_detector()
        self.extractor = extractor or get_embedding_extractor()
        self.matcher = matcher or face_matcher
        self.match_threshold = match_threshold

        logger.info(
            f"Face recognition pipeline initialized with threshold={match_threshold}"
        )

    async def process_image(
        self,
        image: np.ndarray,
        max_faces: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> RecognitionResult:
        """
        Process an image through complete recognition pipeline.

        Steps:
        1. Detect faces in image
        2. Extract embeddings from detected faces
        3. Match embeddings against database
        4. Return consolidated results

        Args:
            image: Input image in BGR format
            max_faces: Maximum number of faces to process
            threshold: Override match threshold

        Returns:
            RecognitionResult with all matches
        """
        start_time = time.time()

        try:
            # Step 1: Detect faces
            logger.debug("Step 1: Detecting faces...")
            detections = self.detector.detect_faces(
                image, align_faces=True, max_faces=max_faces
            )

            if not detections:
                logger.info("No faces detected in image")
                return RecognitionResult(
                    success=True,
                    processing_time=time.time() - start_time,
                    faces_detected=0,
                    faces_matched=0,
                    matches=[],
                )

            logger.info(f"Detected {len(detections)} face(s)")

            # Step 2 & 3: Extract embeddings and match
            logger.debug("Step 2-3: Extracting embeddings and matching...")
            matches = []
            matched_count = 0

            for i, detection in enumerate(detections):
                try:
                    if detection.aligned_face is None:
                        logger.warning(f"Face {i + 1}: No aligned face, skipping")
                        continue

                    # Extract embedding
                    embedding = self.extractor.extract_embedding(
                        detection.aligned_face, normalize=True
                    )

                    # Match against database
                    match_result = await self.matcher.match_face_by_embedding(
                        embedding, threshold=threshold or self.match_threshold
                    )

                    matches.append(match_result)

                    if match_result.is_match:
                        matched_count += 1
                        logger.info(
                            f"Face {i + 1}: Matched to {match_result.employee_id} "
                            f"(confidence: {match_result.confidence:.3f})"
                        )
                    else:
                        logger.info(f"Face {i + 1}: No match found")

                except Exception as e:
                    logger.error(f"Failed to process face {i + 1}: {e}")
                    continue

            processing_time = time.time() - start_time

            logger.info(
                f"Pipeline completed: {matched_count}/{len(detections)} faces matched "
                f"in {processing_time:.3f}s"
            )

            return RecognitionResult(
                success=True,
                processing_time=processing_time,
                faces_detected=len(detections),
                faces_matched=matched_count,
                matches=matches,
            )

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return RecognitionResult(
                success=False,
                processing_time=time.time() - start_time,
                faces_detected=0,
                faces_matched=0,
                matches=[],
                error=str(e),
            )

    async def register_employee_face(
        self,
        employee_id: UUID,
        image: np.ndarray,
        check_duplicate: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RegistrationResult:
        """
        Register employee face through complete pipeline.

        Steps:
        1. Detect face in image
        2. Extract embedding
        3. Check for duplicates (optional)
        4. Store in database

        Args:
            employee_id: UUID of the employee
            image: Face image in BGR format
            check_duplicate: Check if face already registered
            metadata: Additional metadata

        Returns:
            RegistrationResult with registration info
        """
        start_time = time.time()

        try:
            logger.info(f"Registering face for employee {employee_id}")

            # Assess image quality
            from app.ml.face_recognition.preprocessor import preprocessor

            quality_metrics = preprocessor.assess_image_quality(image)
            logger.debug(f"Image quality: {quality_metrics}")

            # Register through matcher (includes detection and extraction)
            result = await self.matcher.register_face(
                employee_id=employee_id,
                face_image=image,
                check_duplicate=check_duplicate,
                metadata=metadata,
            )

            processing_time = time.time() - start_time

            logger.info(
                f"Registration successful for {employee_id} in {processing_time:.3f}s"
            )

            return RegistrationResult(
                success=True,
                employee_id=employee_id,
                point_id=result["point_id"],
                processing_time=processing_time,
                image_quality=quality_metrics,
            )

        except ValueError as e:
            # Expected errors (no face, duplicate, etc.)
            processing_time = time.time() - start_time
            logger.warning(f"Registration failed for {employee_id}: {e}")

            # Still assess quality even on failure
            from app.ml.face_recognition.preprocessor import preprocessor

            quality_metrics = preprocessor.assess_image_quality(image)

            return RegistrationResult(
                success=False,
                employee_id=employee_id,
                point_id=None,
                processing_time=processing_time,
                image_quality=quality_metrics,
                error=str(e),
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Registration error for {employee_id}: {e}")

            return RegistrationResult(
                success=False,
                employee_id=employee_id,
                point_id=None,
                processing_time=processing_time,
                error=f"Internal error: {str(e)}",
            )

    async def verify_employee(
        self,
        employee_id: UUID,
        image: np.ndarray,
        threshold: Optional[float] = None,
    ) -> tuple[bool, float]:
        """
        Verify if image matches a specific employee.

        Args:
            employee_id: Expected employee ID
            image: Face image to verify
            threshold: Override match threshold

        Returns:
            Tuple of (is_verified, confidence_score)
        """
        try:
            result = await self.process_image(
                image, max_faces=1, threshold=threshold or self.match_threshold
            )

            if not result.success or not result.matches:
                return False, 0.0

            match = result.matches[0]

            is_verified = match.is_match and match.employee_id == employee_id
            confidence = match.confidence if match.is_match else 0.0

            logger.info(
                f"Verification for {employee_id}: "
                f"{'PASS' if is_verified else 'FAIL'} ({confidence:.3f})"
            )

            return is_verified, confidence

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False, 0.0

    async def identify_employee(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Optional[tuple[UUID, float]]:
        """
        Identify employee from image (single face expected).

        Args:
            image: Face image in BGR format
            threshold: Override match threshold

        Returns:
            Tuple of (employee_id, confidence) or None if no match
        """
        try:
            result = await self.process_image(
                image, max_faces=1, threshold=threshold or self.match_threshold
            )

            if not result.success or not result.matches:
                return None

            match = result.matches[0]

            if match.is_match and match.employee_id:
                logger.info(
                    f"Identified employee {match.employee_id} "
                    f"(confidence: {match.confidence:.3f})"
                )
                return match.employee_id, match.confidence

            return None

        except Exception as e:
            logger.error(f"Identification failed: {e}")
            return None

    async def process_image_from_bytes(
        self,
        image_bytes: bytes,
        max_faces: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> RecognitionResult:
        """
        Process image from raw bytes.

        Args:
            image_bytes: Image data as bytes
            max_faces: Maximum faces to process
            threshold: Override match threshold

        Returns:
            RecognitionResult
        """
        try:
            # Decode image
            image = preprocessor.decode_image(image_bytes)

            # Process through pipeline
            return await self.process_image(
                image, max_faces=max_faces, threshold=threshold
            )

        except ValueError as e:
            logger.error(f"Failed to decode image: {e}")
            return RecognitionResult(
                success=False,
                processing_time=0.0,
                faces_detected=0,
                faces_matched=0,
                matches=[],
                error=f"Image decode error: {str(e)}",
            )


# Global pipeline instance
face_recognition_pipeline = FaceRecognitionPipeline(
    match_threshold=settings.face_match_threshold
)
