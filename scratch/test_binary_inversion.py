import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import crop_to_content, resize_with_padding
from core.model_inference import get_recognizer

# Create dark mode image: 600x300 white canvas, 400x200 dark rectangle in center, white text "2y+2=12"
img = Image.new("RGB", (600, 300), "white")
draw = ImageDraw.Draw(img)
draw.rectangle([100, 50, 500, 250], fill="#1E1E1E") # dark background

try:
    font = ImageFont.truetype("arial.ttf", 60)
except Exception:
    font = ImageFont.load_default()

draw.text((150, 110), "2y+2=12", fill="white", font=font)

import io
buf = io.BytesIO()
img.save(buf, format="PNG")

bgr = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Threshold with Otsu
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Crop the dark box / content first
cropped_bin = crop_to_content(binary, padding=20)

# Check if cropped_bin has dark background (majority of pixels are black 0)
black_ratio = (cropped_bin == 0).sum() / cropped_bin.size
print("Black pixels ratio in cropped content:", black_ratio)

if black_ratio > 0.5:
    print("Inverting binary image so text becomes black on white!")
    cropped_bin = cv2.bitwise_not(cropped_bin)

# Crop again to get tight bounding box around the black text ink!
tight_cropped = crop_to_content(cropped_bin, padding=20)
resized = resize_with_padding(tight_cropped, 224, 224)

pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))
pil_img.save("scratch/dark_mode_perfect_inversion.png")

recognizer = get_recognizer()
rec = recognizer.recognize(pil_img, num_samples=3)
print("Recognized LaTeX (perfect inversion):", rec.latex)
