"""Face detection using RetinaFace/SCRFD models."""

import logging
from typing import List, Tuple, Optional
import numpy as np

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
except ImportError:
    FaceAnalysis = None
    get_model = None

from app.ml.face_recognition.preprocessor import preprocessor
from app.core.config import settings

logger = logging.getLogger(__name__)


class FaceDetectionResult:
    """Container for face detection results."""

    def __init__(
        self,
        bbox: np.ndarray,
        landmarks: np.ndarray,
        confidence: float,
        aligned_face: Optional[np.ndarray] = None,
    ):
        """
        Initialize detection result.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            landmarks: Facial landmarks (5 points)
            confidence: Detection confidence score
            aligned_face: Optional aligned face image
        """
        self.bbox = bbox
        self.landmarks = landmarks
        self.confidence = confidence
        self.aligned_face = aligned_face

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "bbox": self.bbox.tolist(),
            "landmarks": self.landmarks.tolist(),
            "confidence": float(self.confidence),
        }


class FaceDetector:
    """Face detector using InsightFace (RetinaFace/SCRFD)."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        confidence_threshold: float = 0.8,
        device: str = "cpu",
    ):
        """
        Initialize face detector.

        Args:
            model_name: InsightFace model pack name
            confidence_threshold: Minimum confidence score for detections
            device: Device to run on ('cpu' or 'cuda')

        Raises:
            ImportError: If insightface is not installed
            RuntimeError: If model initialization fails
        """
        if FaceAnalysis is None:
            raise ImportError(
                "insightface is not installed. Install with: pip install insightface"
            )

        self.confidence_threshold = confidence_threshold
        self.device = device

        try:
            # Initialize FaceAnalysis with detection only
            self.app = FaceAnalysis(
                name=model_name,
                providers=["CPUExecutionProvider"]
                if device == "cpu"
                else ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

            # Prepare model (download if needed)
            ctx_id = -1 if device == "cpu" else 0
            det_size = (settings.face_detection_size, settings.face_detection_size)
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)

            logger.info(
                f"Face detector initialized: {model_name} on {device}, "
                f"threshold={confidence_threshold}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            raise RuntimeError(f"Face detector initialization failed: {e}") from e

    def detect_faces(
        self,
        image: np.ndarray,
        align_faces: bool = True,
        max_faces: Optional[int] = None,
    ) -> List[FaceDetectionResult]:
        """
        Detect faces in an image.

        Args:
            image: Input image in BGR format
            align_faces: Whether to align detected faces
            max_faces: Maximum number of faces to detect (None for all)

        Returns:
            List of FaceDetectionResult objects, sorted by confidence (descending)

        Raises:
            ValueError: If image is invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        try:
            # Apply preprocessing if configured
            processed_image = image
            if settings.image_preprocessing != "none":
                logger.debug(f"Applying {settings.image_preprocessing} preprocessing")
                processed_image = preprocessor.apply_pipeline(
                    image, settings.image_preprocessing
                )

            # Run detection
            faces = self.app.get(processed_image, max_num=max_faces or 0)
            
            logger.warning(
                f"InsightFace detected {len(faces)} face(s) before threshold filtering "
                f"(threshold: {self.confidence_threshold})"
            )
            
            # Log all detected faces with their scores
            if faces:
                for i, face in enumerate(faces):
                    logger.warning(
                        f"Face {i+1}: confidence={face.det_score:.4f}, "
                        f"bbox={face.bbox.tolist() if hasattr(face.bbox, 'tolist') else face.bbox}"
                    )
            else:
                logger.warning("InsightFace did not detect any faces at all!")

            # Filter by confidence and convert to results
            results = []
            filtered_count = 0
            for face in faces:
                if face.det_score >= self.confidence_threshold:
                    # Get aligned face if requested
                    aligned = None
                    if align_faces and face.kps is not None:
                        aligned = preprocessor.align_face(
                            image,
                            face.kps,
                            output_size=(112, 112),
                        )

                    result = FaceDetectionResult(
                        bbox=face.bbox,
                        landmarks=face.kps,
                        confidence=face.det_score,
                        aligned_face=aligned,
                    )
                    results.append(result)
                else:
                    filtered_count += 1
                    logger.warning(
                        f"Face filtered out: confidence={face.det_score:.4f} < threshold={self.confidence_threshold}"
                    )

            # Sort by confidence (descending)
            results.sort(key=lambda x: x.confidence, reverse=True)

            logger.warning(
                f"After filtering: {len(results)} faces passed, {filtered_count} faces filtered out "
                f"(threshold: {self.confidence_threshold})"
            )

            return results

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            raise RuntimeError(f"Face detection error: {e}") from e

    def detect_single_face(
        self,
        image: np.ndarray,
        align_face: bool = True,
    ) -> Optional[FaceDetectionResult]:
        """
        Detect single face with highest confidence.

        Args:
            image: Input image in BGR format
            align_face: Whether to align the detected face

        Returns:
            FaceDetectionResult if face found, None otherwise
        """
        results = self.detect_faces(image, align_faces=align_face, max_faces=1)
        return results[0] if results else None

    def detect_and_crop_faces(
        self,
        image: np.ndarray,
        margin: float = 0.1,
        max_faces: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, FaceDetectionResult]]:
        """
        Detect faces and return cropped face images.

        Args:
            image: Input image in BGR format
            margin: Margin percentage around bbox (0.0-1.0)
            max_faces: Maximum number of faces to detect

        Returns:
            List of (cropped_face, detection_result) tuples
        """
        results = self.detect_faces(image, align_faces=False, max_faces=max_faces)

        cropped_faces = []
        for result in results:
            cropped = preprocessor.crop_face(image, result.bbox, margin=margin)
            cropped_faces.append((cropped, result))

        return cropped_faces

    def batch_detect_faces(
        self,
        images: List[np.ndarray],
        align_faces: bool = True,
        max_faces_per_image: Optional[int] = None,
    ) -> List[List[FaceDetectionResult]]:
        """
        Batch detect faces in multiple images.

        Args:
            images: List of input images in BGR format
            align_faces: Whether to align detected faces
            max_faces_per_image: Max faces per image

        Returns:
            List of detection results for each image
        """
        all_results = []
        for image in images:
            try:
                results = self.detect_faces(
                    image,
                    align_faces=align_faces,
                    max_faces=max_faces_per_image,
                )
                all_results.append(results)
            except Exception as e:
                logger.warning(f"Failed to detect faces in batch image: {e}")
                all_results.append([])

        return all_results


# Global detector instance (lazy initialization)
_detector: Optional[FaceDetector] = None


def get_face_detector(
    confidence_threshold: Optional[float] = None,
) -> FaceDetector:
    """
    Get or create global face detector instance.

    Args:
        confidence_threshold: Optional override for confidence threshold

    Returns:
        FaceDetector instance
    """
    global _detector

    threshold = confidence_threshold or settings.face_detection_threshold
    device = "cuda" if settings.use_gpu else "cpu"

    if _detector is None:
        _detector = FaceDetector(
            model_name=settings.insightface_model_pack,
            confidence_threshold=threshold,
            device=device,
        )
    elif confidence_threshold is not None:
        _detector.confidence_threshold = threshold

    return _detector
