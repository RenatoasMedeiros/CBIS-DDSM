 
import subprocess

scripts = [
    'EfficientNetB0_224_32.py',
    'EfficientNetB0_224_64.py',
    'EfficientNetB0_224_128_512_256_128.py',
    'EfficientNetB4_224_32_64_02.py',
    'EfficientNetB4_224_32.py',
    'EfficientNetB4_224_64.py',
    'EfficientNetB4_224_128_512_256_128.py',
    'EfficientNetB7_224_32_512_256_128.py',
    'EfficientNetB7_224_64_512_256_128.py',
]

for script in scripts:
    try:
        subprocess.run([".venv/bin/python", script])
    except Exception as e:
        print(f"Error occurred while executing {script}: {e}")