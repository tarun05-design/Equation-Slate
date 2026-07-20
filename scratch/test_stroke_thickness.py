import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import crop_to_content, resize_with_padding
from core.model_inference import get_recognizer

# Load exact user input
bgr = cv2.imread("scratch/latest_input.png")
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

cropped = crop_to_content(binary, padding=20)
print("Cropped shape (h, w):", cropped.shape)

recognizer = get_recognizer()

# Test 1: Standard resize (224x224)
resized_std = resize_with_padding(cropped, 224, 224)
pil_std = Image.fromarray(cv2.cvtColor(resized_std, cv2.COLOR_GRAY2RGB))
rec_std = recognizer.recognize(pil_std, num_samples=3)
print("Standard (224x224) prediction:", rec_std.latex)

# Test 2: Thicker stroke (Dilation)
kernel = np.ones((3, 3), np.uint8)
# binary is black ink (0) on white (255), so erosion on binary thickens black ink
thickened = cv2.erode(cropped, kernel, iterations=1)
resized_thick = resize_with_padding(thickened, 224, 224)
pil_thick = Image.fromarray(cv2.cvtColor(resized_thick, cv2.COLOR_GRAY2RGB))
rec_thick = recognizer.recognize(pil_thick, num_samples=3)
print("Thickened stroke prediction:", rec_thick.latex)

# Test 3: Pix2Tex native size or dynamic scaling (e.g. 384x120 or keeping height ~120)
# Pix2Tex handles arbitrary width/height!
h, w = cropped.shape[:2]
target_h = 120
target_w = max(120, int(w * (target_h / h)))
resized_dynamic = resize_with_padding(cropped, target_w, target_h)
pil_dynamic = Image.fromarray(cv2.cvtColor(resized_dynamic, cv2.COLOR_GRAY2RGB))
rec_dynamic = recognizer.recognize(pil_dynamic, num_samples=3)
print("Dynamic aspect ratio prediction:", rec_dynamic.latex)
