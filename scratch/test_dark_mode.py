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
img_bytes = buf.getvalue()

bgr = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Check dark mode inversion: if average background inside or overall mean < 128
# Or threshold
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# In binary: 0 is black, 255 is white.
# Standard image: background is 255 (white), ink is 0 (black).
# Dark mode image: background is 0 (black), text is 255 (white).
# If count of 0 (black pixels) > count of 255 (white pixels), it's dark mode!
dark_ratio = (binary == 0).sum() / binary.size
print("Dark pixels ratio:", dark_ratio)

if dark_ratio > 0.5:
    print("Inverting dark mode image!")
    binary = cv2.bitwise_not(binary)

cropped = crop_to_content(binary, padding=20)
resized = resize_with_padding(cropped, 224, 224)
pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))
pil_img.save("scratch/dark_mode_inverted.png")

recognizer = get_recognizer()
rec = recognizer.recognize(pil_img, num_samples=3)
print("Recognized LaTeX (after inversion):", rec.latex)
