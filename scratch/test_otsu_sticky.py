import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import crop_to_content, resize_with_padding
from core.model_inference import get_recognizer

# Create sticky note image with 800x400 white border and yellow sticky note in center
img = Image.new("RGB", (800, 400), "white")
draw = ImageDraw.Draw(img)
draw.rectangle([100, 50, 700, 350], fill="#F7E398")

try:
    font = ImageFont.truetype("arial.ttf", 70)
except Exception:
    font = ImageFont.load_default()

draw.text((150, 140), "2y", fill="black", font=font)
draw.text((230, 140), "-3x", fill="red", font=font)
draw.text((360, 140), "=-4", fill="black", font=font)

import io
buf = io.BytesIO()
img.save(buf, format="PNG")

bgr = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Otsu thresholding
_, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("scratch/test_otsu_sticky.png", binary_otsu)

cropped = crop_to_content(binary_otsu, padding=20)
resized = resize_with_padding(cropped, 224, 224)
pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))
pil_img.save("scratch/test_otsu_preprocessed.png")

recognizer = get_recognizer()
rec = recognizer.recognize(pil_img, num_samples=3)
print("Otsu Recognition result:", rec.latex)
