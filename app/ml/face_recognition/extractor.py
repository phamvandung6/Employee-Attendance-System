"""Face embedding extraction using ArcFace model."""

import logging
from typing import List, Optional
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

from app.ml.face_recognition.preprocessor import preprocessor
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Face embedding extractor using ArcFace (InsightFace)."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cpu",
    ):
        """
        Initialize embedding extractor.

        Args:
            model_name: InsightFace model pack name (includes ArcFace)
            device: Device to run on ('cpu' or 'cuda')

        Raises:
            ImportError: If insightface is not installed
            RuntimeError: If model initialization fails
        """
        if FaceAnalysis is None:
            raise ImportError(
                "insightface is not installed. Install with: pip install insightface"
            )

        self.device = device
        self.embedding_size = 512  # ArcFace produces 512-dimensional embeddings

        try:
            # Initialize FaceAnalysis with recognition model
            self.app = FaceAnalysis(
                name=model_name,
                providers=["CPUExecutionProvider"]
                if device == "cpu"
                else ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

            # Prepare model (download if needed)
            ctx_id = -1 if device == "cpu" else 0
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

            logger.info(
                f"Embedding extractor initialized: {model_name} on {device}, "
                f"embedding_size={self.embedding_size}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize embedding extractor: {e}")
            raise RuntimeError(f"Embedding extractor initialization failed: {e}") from e

    def extract_embedding(
        self,
        face_image: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract face embedding from a face image.

        Args:
            face_image: Aligned face image (preferably 112x112) in BGR format
            normalize: Whether to L2-normalize the embedding

        Returns:
            Face embedding vector (512-dimensional)

        Raises:
            ValueError: If face cannot be processed
            RuntimeError: If extraction fails
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Invalid face image")

        try:
            # Check if this is an aligned face (112x112) - if so, extract directly
            h, w = face_image.shape[:2]
            is_aligned_face = h == 112 and w == 112

            if is_aligned_face:
                # For aligned faces, extract embedding directly from recognition model
                # This is faster than creating dummy image and detecting again
                logger.debug(
                    "Extracting embedding directly from aligned face (112x112)"
                )

                # Get recognition model directly
                if "recognition" not in self.app.models:
                    raise RuntimeError("Recognition model not available")

                rec_model = self.app.models["recognition"]

                # Create a minimal face object for aligned face
                # Aligned face is 112x112, so bbox covers entire image
                # Use standard landmarks for aligned face (centered)
                class SimpleFace:
                    def __init__(self):
                        # Bbox covering entire 112x112 image
                        self.bbox = np.array([0.0, 0.0, 112.0, 112.0])
                        # Standard landmarks for aligned face (centered)
                        self.kps = np.array(
                            [
                                [38.2946, 51.6963],  # Left eye
                                [73.5318, 51.5014],  # Right eye
                                [56.0252, 71.7366],  # Nose tip
                                [41.5493, 92.3655],  # Left mouth corner
                                [70.7299, 92.2041],  # Right mouth corner
                            ],
                            dtype=np.float32,
                        )

                face_obj = SimpleFace()

                # Call recognition model with aligned face and face object
                embedding = rec_model.get(face_image, face_obj)

                # Normalize if requested
                if normalize:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    else:
                        raise ValueError("Zero embedding norm")
            else:
                # For full images, detect face first then extract embedding
                logger.debug("Detecting face in full image before extraction")
                faces = self.app.get(face_image)

                if not faces:
                    raise ValueError("No face detected in the provided image")

                # Use the first (and should be only) face
                face = faces[0]

                # Get embedding
                embedding = face.normed_embedding if normalize else face.embedding

                if embedding is None:
                    raise ValueError("Failed to extract embedding from face")

            # Ensure correct shape
            if embedding.shape[0] != self.embedding_size:
                raise ValueError(
                    f"Unexpected embedding size: {embedding.shape[0]}, "
                    f"expected {self.embedding_size}"
                )

            logger.debug(f"Extracted embedding with shape: {embedding.shape}")

            return embedding

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise RuntimeError(f"Embedding extraction error: {e}") from e

    def extract_embedding_from_aligned(
        self,
        aligned_face: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract embedding from pre-aligned face (112x112).

        Args:
            aligned_face: Pre-aligned face image (112x112) in BGR format
            normalize: Whether to L2-normalize the embedding

        Returns:
            Face embedding vector (512-dimensional)
        """
        # Aligned face should be 112x112
        if aligned_face.shape[:2] != (112, 112):
            # Resize if needed
            aligned_face = preprocessor.resize_image(
                aligned_face, max_size=112, keep_aspect_ratio=False
            )[0]

        return self.extract_embedding(aligned_face, normalize=normalize)

    def batch_extract_embeddings(
        self,
        face_images: List[np.ndarray],
        normalize: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings from multiple face images.

        Args:
            face_images: List of face images in BGR format
            normalize: Whether to L2-normalize the embeddings

        Returns:
            List of embeddings (None for failed extractions)
        """
        embeddings = []

        for face_image in face_images:
            try:
                embedding = self.extract_embedding(face_image, normalize=normalize)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to extract embedding from batch image: {e}")
                embeddings.append(None)

        logger.info(
            f"Extracted {sum(1 for e in embeddings if e is not None)}/{len(face_images)} "
            f"embeddings from batch"
        )

        return embeddings

    @staticmethod
    def compute_similarity(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine",
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine' or 'euclidean')

        Returns:
            Similarity score (higher = more similar)

        Raises:
            ValueError: If embeddings have different shapes or invalid metric
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError(
                f"Embedding shapes don't match: {embedding1.shape} vs {embedding2.shape}"
            )

        if metric == "cosine":
            # Cosine similarity (assumes embeddings are already normalized)
            similarity = float(np.dot(embedding1, embedding2))
            # Ensure in [-1, 1] range
            similarity = np.clip(similarity, -1.0, 1.0)
            return similarity

        elif metric == "euclidean":
            # Euclidean distance (convert to similarity score)
            distance = float(np.linalg.norm(embedding1 - embedding2))
            # Convert distance to similarity (inverse relationship)
            similarity = 1.0 / (1.0 + distance)
            return similarity

        else:
            raise ValueError(f"Invalid metric: {metric}. Use 'cosine' or 'euclidean'")

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        L2-normalize an embedding vector.

        Args:
            embedding: Input embedding vector

        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def verify_faces(
        self,
        face1: np.ndarray,
        face2: np.ndarray,
        threshold: float = 0.6,
    ) -> tuple[bool, float]:
        """
        Verify if two face images belong to the same person.

        Args:
            face1: First face image in BGR format
            face2: Second face image in BGR format
            threshold: Similarity threshold for verification

        Returns:
            Tuple of (is_same_person, similarity_score)
        """
        try:
            embedding1 = self.extract_embedding(face1, normalize=True)
            embedding2 = self.extract_embedding(face2, normalize=True)

            similarity = self.compute_similarity(
                embedding1, embedding2, metric="cosine"
            )
            is_same = similarity >= threshold

            logger.debug(
                f"Face verification: similarity={similarity:.4f}, "
                f"threshold={threshold}, match={is_same}"
            )

            return is_same, float(similarity)

        except Exception as e:
            logger.error(f"Face verification failed: {e}")
            raise


# Global extractor instance (lazy initialization)
_extractor: Optional[EmbeddingExtractor] = None


def get_embedding_extractor() -> EmbeddingExtractor:
    """
    Get or create global embedding extractor instance.

    Returns:
        EmbeddingExtractor instance
    """
    global _extractor

    if _extractor is None:
        device = "cuda" if settings.use_gpu else "cpu"
        _extractor = EmbeddingExtractor(
            model_name=settings.insightface_model_pack,
            device=device,
        )

    return _extractor
