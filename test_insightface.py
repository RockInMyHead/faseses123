#!/usr/bin/env python3
"""Тест для проверки работы insightface"""

import sys
import traceback

try:
    print("Testing insightface import...")
    import insightface
    print(f"✅ InsightFace version: {insightface.__version__}")

    print("Testing FaceAnalysis import...")
    from insightface.app import FaceAnalysis
    print("✅ FaceAnalysis imported successfully")

    print("Testing FaceAnalysis initialization...")
    app = FaceAnalysis(name="buffalo_l")
    print("✅ FaceAnalysis initialized")

    print("Testing prepare() method...")
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ FaceAnalysis prepared successfully")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)

print("🎉 All tests passed!")
