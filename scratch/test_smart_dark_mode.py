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
_, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Crop the region of interest (non-white outer margins)
# In gray, background inside dark card is ~30, outer border is 255.
# Let's find the inner dark card or math content.
# If average intensity of pixels inside the card/image is dark (< 128), it's dark mode!
card_mask = gray < 200
card_gray = gray[card_mask]
print("Card region mean gray intensity:", card_gray.mean())

# If card_gray.mean() < 128, the background is dark and text is light!
if card_gray.mean() < 128:
    print("Detected Dark Mode! Inverting card region...")
    # Invert the dark card
    gray_inv = gray.copy()
    gray_inv[card_mask] = 255 - gray[card_mask]
    _, binary_final = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
else:
    binary_final = binary_otsu

cropped = crop_to_content(binary_final, padding=20)
resized = resize_with_padding(cropped, 224, 224)
pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))
pil_img.save("scratch/dark_mode_correct_inversion.png")

recognizer = get_recognizer()
rec = recognizer.recognize(pil_img, num_samples=3)
print("Recognized LaTeX (with smart dark mode inversion):", rec.latex)
