#!/usr/bin/env python3
"""Simple script to test face detection and registration."""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.ml.face_recognition.detector import get_face_detector
from app.ml.face_recognition.extractor import get_embedding_extractor
from app.ml.face_recognition.preprocessor import preprocessor
from app.core.config import settings

def test_face_detection(image_path: str):
    """Test face detection on an image."""
    print(f"Testing face detection on: {image_path}")
    print(f"Detection threshold: {settings.face_detection_threshold}")
    print("-" * 60)
    
    # Read image
    print("1. Reading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read image from {image_path}")
        return False
    
    print(f"   ✓ Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"   ✓ Image min: {image.min()}, max: {image.max()}")
    
    # Resize if too large
    h, w = image.shape[:2]
    if max(h, w) > 1280:
        print(f"2. Resizing image from {w}x{h}...")
        resized, scale = preprocessor.resize_image(image, max_size=1280, keep_aspect_ratio=True)
        image = resized
        print(f"   ✓ Resized to: {image.shape}")
    else:
        print("2. Image size OK, no resize needed")
    
    # Get detector
    print("3. Initializing detector...")
    detector = get_face_detector()
    print(f"   ✓ Detector initialized with threshold: {detector.confidence_threshold}")
    
    # Detect faces
    print("4. Detecting faces...")
    try:
        detections = detector.detect_faces(image, align_faces=True, max_faces=1)
        print(f"   ✓ Detected {len(detections)} face(s)")
        
        if not detections:
            print("   ❌ No faces detected!")
            return False
        
        for i, detection in enumerate(detections):
            print(f"\n   Face {i+1}:")
            print(f"     - Confidence: {detection.confidence:.4f}")
            print(f"     - Bbox: {detection.bbox}")
            print(f"     - Has aligned face: {detection.aligned_face is not None}")
            if detection.aligned_face is not None:
                print(f"     - Aligned face shape: {detection.aligned_face.shape}")
        
        # Test detect_single_face
        print("\n5. Testing detect_single_face...")
        single_detection = detector.detect_single_face(image, align_face=True)
        if single_detection is None:
            print("   ❌ detect_single_face returned None!")
            print("   This is the problem!")
            return False
        else:
            print(f"   ✓ detect_single_face returned face with confidence: {single_detection.confidence:.4f}")
            print(f"   ✓ Has aligned face: {single_detection.aligned_face is not None}")
        
        # Test embedding extraction
        if single_detection.aligned_face is not None:
            print("\n6. Testing embedding extraction...")
            extractor = get_embedding_extractor()
            embedding = extractor.extract_embedding(single_detection.aligned_face, normalize=True)
            print(f"   ✓ Embedding extracted: shape={embedding.shape}, dtype={embedding.dtype}")
            print(f"   ✓ Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    image_path = "/home/yanaa/Documents/image_test/register.jpg"
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    success = test_face_detection(image_path)
    sys.exit(0 if success else 1)

