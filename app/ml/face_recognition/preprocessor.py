"""Image preprocessing utilities for face detection and recognition."""

import cv2
import numpy as np
from typing import Tuple


class ImagePreprocessor:
    """Handles image preprocessing for face detection and recognition."""

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
