#!/usr/bin/env python3
"""Проверка установленных библиотек"""

import sys
print('Python path:', sys.executable)
print('Python version:', sys.version)

try:
    import face_recognition
    print('✅ face_recognition: OK')
    print('   Version:', getattr(face_recognition, '__version__', 'unknown'))
except ImportError as e:
    print('❌ face_recognition: FAILED -', e)

try:
    import insightface
    print('✅ insightface: OK')
    print('   Version:', getattr(insightface, '__version__', 'unknown'))
except ImportError as e:
    print('❌ insightface: FAILED -', e)

try:
    import onnxruntime
    print('✅ onnxruntime: OK')
    print('   Version:', getattr(onnxruntime, '__version__', 'unknown'))
except ImportError as e:
    print('❌ onnxruntime: FAILED -', e)

try:
    import dlib
    print('✅ dlib: OK')
    print('   Version:', getattr(dlib, '__version__', 'unknown'))
except ImportError as e:
    print('❌ dlib: FAILED -', e)
