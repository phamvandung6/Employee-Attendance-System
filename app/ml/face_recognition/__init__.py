"""Face recognition module."""

from app.ml.face_recognition.preprocessor import ImagePreprocessor, preprocessor
from app.ml.face_recognition.detector import (
    FaceDetector,
    FaceDetectionResult,
    get_face_detector,
)
from app.ml.face_recognition.extractor import (
    EmbeddingExtractor,
    get_embedding_extractor,
)
from app.ml.face_recognition.matcher import (
    FaceMatcher,
    FaceMatchResult,
    face_matcher,
)
from app.ml.face_recognition.pipeline import (
    FaceRecognitionPipeline,
    RecognitionResult,
    RegistrationResult,
    face_recognition_pipeline,
)

__all__ = [
    "ImagePreprocessor",
    "preprocessor",
    "FaceDetector",
    "FaceDetectionResult",
    "get_face_detector",
    "EmbeddingExtractor",
    "get_embedding_extractor",
    "FaceMatcher",
    "FaceMatchResult",
    "face_matcher",
    "FaceRecognitionPipeline",
    "RecognitionResult",
    "RegistrationResult",
    "face_recognition_pipeline",
]
