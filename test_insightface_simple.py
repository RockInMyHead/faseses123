#!/usr/bin/env python3
"""Простой тест insightface"""

import sys
import traceback

try:
    print("Testing insightface...")

    from insightface.app import FaceAnalysis
    print("FaceAnalysis imported successfully")

    app = FaceAnalysis(name="buffalo_l")
    print("FaceAnalysis initialized")

    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model prepared")

    # Тест на простом изображении
    from pathlib import Path
    test_dir = Path("test_photos")
    if test_dir.exists():
        test_img = next(test_dir.glob("*.jpg"), None)
        if test_img:
            from PIL import Image
            import numpy as np

            print(f"Testing on: {test_img.name}")
            image = Image.open(test_img)
            image_np = np.array(image)

            faces = app.get(image_np)
            print(f"Found faces: {len(faces)}")

            for i, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                print(f"   Face {i+1}: bbox={bbox.tolist()}, confidence={face.det_score:.3f}")

    print("InsightFace works perfectly!")

except Exception as e:
    print(f"Error: {e}")
    print("Details:")
    traceback.print_exc()
    sys.exit(1)
