"""Image preprocessing utilities for face detection and recognition."""

import cv2
import numpy as np
from typing import Tuple, Literal
import logging

logger = logging.getLogger(__name__)

PreprocessPipeline = Literal["none", "standard", "quality"]


class ImagePreprocessor:
    """Handles image preprocessing for face detection and recognition."""

    @staticmethod
    def apply_pipeline(
        image: np.ndarray,
        pipeline: PreprocessPipeline = "none",
    ) -> np.ndarray:
        """
        Apply preprocessing pipeline to image.

        Args:
            image: Input BGR image
            pipeline: Type of preprocessing pipeline:
                - "none": No preprocessing (return original)
                - "standard": CLAHE only (balanced)
                - "quality": Denoise + CLAHE + Sharpen (for poor quality images)

        Returns:
            Preprocessed BGR image

        Note:
            Modern face recognition models (InsightFace, ArcFace) are trained on
            diverse raw images and typically don't need preprocessing. Use preprocessing
            only when dealing with poor quality images (mobile cameras, CCTV, low light).

        Performance Impact:
            - none: ~0ms
            - standard: ~15ms (+30%)
            - quality: ~50ms (+100%)
        """
        if pipeline == "none":
            return image

        processed = image.copy()

        if pipeline == "quality":
            # Full pipeline for poor quality images
            logger.debug("Applying quality preprocessing pipeline")

            # Step 1: Denoise with bilateral filter (preserves edges)
            processed = cv2.bilateralFilter(processed, 9, 75, 75)

            # Step 2: CLAHE for contrast enhancement
            processed = ImagePreprocessor._apply_clahe(
                processed, clip_limit=3.0, tile_grid_size=(8, 8)
            )

            # Step 3: Slight sharpening
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            processed = cv2.filter2D(processed, -1, kernel)

        elif pipeline == "standard":
            # Lightweight CLAHE only
            logger.debug("Applying standard preprocessing pipeline")
            processed = ImagePreprocessor._apply_clahe(
                processed, clip_limit=2.0, tile_grid_size=(8, 8)
            )

        return processed

    @staticmethod
    def _apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to BGR image.

        Args:
            image: Input BGR image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization

        Returns:
            CLAHE-enhanced BGR image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)

        # Merge back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def assess_image_quality(image: np.ndarray) -> dict:
        """
        Assess image quality metrics to determine if preprocessing is needed.

        Args:
            image: Input BGR image

        Returns:
            Dictionary with quality metrics:
                - brightness: Mean pixel intensity (0-255)
                - contrast: Standard deviation of intensity
                - blur_score: Laplacian variance (>100 is sharp)
                - needs_preprocessing: Boolean recommendation

        Example:
            >>> quality = preprocessor.assess_image_quality(img)
            >>> if quality['needs_preprocessing']:
            >>>     img = preprocessor.apply_pipeline(img, 'standard')
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate metrics
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Determine if preprocessing is needed
        needs_preprocessing = (
            brightness < 60  # Too dark
            or brightness > 200  # Too bright
            or contrast < 30  # Low contrast
            or blur_score < 100  # Blurry
        )

        return {
            "brightness": brightness,
            "contrast": contrast,
            "blur_score": blur_score,
            "needs_preprocessing": needs_preprocessing,
            "quality_score": min(
                100, (blur_score / 10 + contrast + (100 - abs(128 - brightness))) / 3
            ),
        }

    @staticmethod
    def resize_image(
        image: np.ndarray,
        max_size: int = 1024,
        keep_aspect_ratio: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Resize image to maximum dimension while optionally keeping aspect ratio.

        Args:
            image: Input image (BGR format)
            max_size: Maximum dimension (width or height)
            keep_aspect_ratio: Whether to maintain aspect ratio

        Returns:
            Tuple of (resized_image, scale_factor)
        """
        h, w = image.shape[:2]

        if keep_aspect_ratio:
            # Calculate scale factor
            scale = min(max_size / w, max_size / h)
            if scale >= 1.0:
                # Image is already smaller than max_size
                return image, 1.0

            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            scale = max_size / max(w, h)
            new_w = max_size if w > h else int(w * scale)
            new_h = max_size if h <= w else int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale

    @staticmethod
    def normalize_image(
        image: np.ndarray,
        mean: Tuple[float, float, float] = (127.5, 127.5, 127.5),
        std: Tuple[float, float, float] = (128.0, 128.0, 128.0),
    ) -> np.ndarray:
        """
        Normalize image to [-1, 1] range.

        Args:
            image: Input image (BGR format)
            mean: Mean values for normalization
            std: Standard deviation for normalization

        Returns:
            Normalized image
        """
        normalized = image.astype(np.float32)
        normalized = (normalized - mean) / std
        return normalized

    @staticmethod
    def convert_to_rgb(image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to RGB.

        Args:
            image: Input image in BGR format

        Returns:
            Image in RGB format
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def convert_to_bgr(image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to BGR.

        Args:
            image: Input image in RGB format

        Returns:
            Image in BGR format
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def read_image(file_path: str) -> np.ndarray:
        """
        Read image from file path.

        Args:
            file_path: Path to image file

        Returns:
            Image in BGR format

        Raises:
            ValueError: If image cannot be read
        """
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to read image from: {file_path}")
        return image

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        """
        Decode image from bytes.

        Args:
            image_bytes: Image data as bytes

        Returns:
            Image in BGR format

        Raises:
            ValueError: If image cannot be decoded
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from bytes")
        return image

    @staticmethod
    def encode_image(
        image: np.ndarray,
        format: str = ".jpg",
        quality: int = 95,
    ) -> bytes:
        """
        Encode image to bytes.

        Args:
            image: Input image (BGR format)
            format: Output format (e.g., '.jpg', '.png')
            quality: JPEG quality (0-100)

        Returns:
            Encoded image bytes

        Raises:
            ValueError: If encoding fails
        """
        params = []
        if format.lower() in [".jpg", ".jpeg"]:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif format.lower() == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 11)]

        success, encoded = cv2.imencode(format, image, params)
        if not success:
            raise ValueError(f"Failed to encode image to {format}")

        return encoded.tobytes()

    @staticmethod
    def crop_face(
        image: np.ndarray,
        bbox: np.ndarray,
        margin: float = 0.0,
    ) -> np.ndarray:
        """
        Crop face region from image with optional margin.

        Args:
            image: Input image (BGR format)
            bbox: Bounding box [x1, y1, x2, y2]
            margin: Margin percentage to add around bbox (0.0-1.0)

        Returns:
            Cropped face image
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        if margin > 0:
            # Calculate margin in pixels
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            margin_w = int(bbox_w * margin)
            margin_h = int(bbox_h * margin)

            # Expand bbox with margin
            x1 = max(0, x1 - margin_w)
            y1 = max(0, y1 - margin_h)
            x2 = min(w, x2 + margin_w)
            y2 = min(h, y2 + margin_h)

        # Ensure coordinates are integers and within bounds
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        return image[y1:y2, x1:x2]

    @staticmethod
    def align_face(
        image: np.ndarray,
        landmarks: np.ndarray,
        output_size: Tuple[int, int] = (112, 112),
    ) -> np.ndarray:
        """
        Align face using facial landmarks (eyes, nose, mouth).

        Args:
            image: Input image (BGR format)
            landmarks: Facial landmarks array (5 points: 2 eyes, nose, 2 mouth corners)
            output_size: Output face size (width, height)

        Returns:
            Aligned face image
        """
        # Standard face template (normalized coordinates)
        src = np.array(
            [
                [38.2946, 51.6963],  # Left eye
                [73.5318, 51.5014],  # Right eye
                [56.0252, 71.7366],  # Nose tip
                [41.5493, 92.3655],  # Left mouth corner
                [70.7299, 92.2041],  # Right mouth corner
            ],
            dtype=np.float32,
        )

        # Scale template to output size
        src[:, 0] = src[:, 0] * output_size[0] / 112
        src[:, 1] = src[:, 1] * output_size[1] / 112

        # Calculate affine transformation
        tform = cv2.estimateAffinePartial2D(landmarks, src)[0]

        # Apply transformation
        aligned = cv2.warpAffine(
            image,
            tform,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return aligned


# Global preprocessor instance
preprocessor = ImagePreprocessor()
