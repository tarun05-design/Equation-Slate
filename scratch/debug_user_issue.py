import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import preprocess_image, threshold, crop_to_content
from core.model_inference import get_recognizer

# Create 800x360 canvas with #FFFFFC background (same as app.js)
img = Image.new("RGB", (800, 360), "#FFFFFC")
draw = ImageDraw.Draw(img)

# Draw 2 + 2 similar to screenshot
try:
    font = ImageFont.truetype("arial.ttf", 90)
    draw.text((150, 100), "2 + 2", fill="#23283A", font=font)
except Exception:
    draw.text((150, 100), "2 + 2", fill="#23283A")

# Add a tiny noise pixel at top left (0,0) or edge to test if adaptive threshold / crop fails
# In adaptive thresholding, edge pixels often get binarized to 0
gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
bin_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)

print("Adaptive threshold ink count:", np.count_nonzero(bin_adaptive < 128))

# Check bounding box of adaptive vs simple threshold
coords_adapt = cv2.findNonZero((bin_adaptive < 128).astype(np.uint8))
print("Adaptive bounding box (x, y, w, h):", cv2.boundingRect(coords_adapt))

# Simple threshold (e.g. threshold at 200)
_, bin_simple = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
coords_simple = cv2.findNonZero((bin_simple < 128).astype(np.uint8))
print("Simple threshold bounding box (x, y, w, h):", cv2.boundingRect(coords_simple))

# Test Pix2Tex on cropped simple vs cropped adaptive
cropped_simple = crop_to_content(bin_simple)
from core.preprocessing import resize_with_padding
resized_simple = resize_with_padding(cropped_simple, 224, 224)
pil_simple = Image.fromarray(cv2.cvtColor(resized_simple, cv2.COLOR_GRAY2RGB))

recognizer = get_recognizer()
rec_simple = recognizer.recognize(pil_simple, num_samples=1)
print("Recognized (simple threshold):", rec_simple.latex)
