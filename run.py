 
import subprocess

scripts = [
    "ModelS_244_64_BC_A_Data_Augmentation.py",
    "ModelS_244_64_BC_A_No_DataAugmentation.py",
    "ModelS_244_256_BC_A_Data_Augmentation.py",
    "ModelS_244_256_BC_A_No_DataAugmentation.py",
    "EfficientNetB0_244_64_BinaryCrossentropy_Adam.py",
    "EfficientNetB0_244_64_BinaryCrossentropy_SGD_With_Momentum.py",
    "EfficientNetB0_244_64_BinaryCrossentropy.py",
    "EfficientNetB0_244_64_Dropout_L2.py",
    "EfficientNetB0_244_64_HingeLoss.py",
    "EfficientNetB0_244_256_Dropout_L2.py",
    "EfficientNetB0_244_256_HingeLoss.py",
    "EfficientNetB0_244_256_BinaryCrossentropy.py",
    "EfficientNetB0_244_256_BinaryCrossentropy_SGD_With_Momentum.py",
    "EfficientNetB0_244_256_BinaryCrossentropy_Adam.py",
    "EfficientNetB0_224_64.py",
    "EfficientNetB0_224_128_512_256_128.py",
    "EfficientNetB0_224_32..py",
    "EfficientNetB4_224_32_64_02.py",
    "EfficientNetB4_224_32.py",
    "EfficientNetB4_224_64.py",
    "EfficientNetB4_224_128_512_256_128.py",
    "EfficientNetB7_224_32_512_256_128.py",
    ]

for script in scripts:
    try:
        subprocess.run([".venv/bin/python", script])
    except Exception as e:
        print(f"Error occurred while executing {script}: {e}")