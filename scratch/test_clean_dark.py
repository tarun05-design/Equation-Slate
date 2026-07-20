import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import crop_to_content, resize_with_padding
from core.model_inference import get_recognizer

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

# Find dark background regions (e.g. gray < 100)
dark_mask = gray < 100
# Invert dark regions: dark background (30 -> 225), white text (255 -> 0)
gray_clean = gray.copy()
gray_clean[dark_mask] = 255 - gray[dark_mask]

# Also invert any bright text pixels that were inside the dark card
# In original gray, text pixels inside dark card are > 200, surrounded by < 100
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
dark_dilated = cv2.dilate(dark_mask.astype(np.uint8), kernel)
# Text pixels inside the dilated dark card region:
text_in_card = (dark_dilated == 1) & (gray > 150)
gray_clean[text_in_card] = 255 - gray[text_in_card]

_, binary = cv2.threshold(gray_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cropped = crop_to_content(binary, padding=20)
resized = resize_with_padding(cropped, 224, 224)
pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))
pil_img.save("scratch/dark_mode_inverted_clean.png")

recognizer = get_recognizer()
rec = recognizer.recognize(pil_img, num_samples=3)
print("Recognized LaTeX (clean dark mode inversion):", rec.latex)
