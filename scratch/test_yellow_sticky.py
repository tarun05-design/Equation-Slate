import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import preprocess_image
from core.model_inference import get_recognizer

# Recreate yellow sticky note on white padding
img = Image.new("RGB", (600, 300), "white")
draw = ImageDraw.Draw(img)

# Draw yellow sticky note patch in center
draw.rectangle([60, 20, 540, 280], fill="#F9E79F")

# Draw "2y - 3x = -4" with red y and x, black 2, -3, =-4
try:
    font = ImageFont.truetype("arial.ttf", 90)
except Exception:
    font = ImageFont.load_default()

# 2 (black), y (red), -3 (black), x (red), = -4 (black)
draw.text((90, 80), "2", fill="black", font=font)
draw.text((150, 80), "y", fill="red", font=font)
draw.text((220, 80), " - 3", fill="black", font=font)
draw.text((360, 80), "x", fill="red", font=font)
draw.text((420, 80), " = -4", fill="black", font=font)

import io
buf = io.BytesIO()
img.save(buf, format="PNG")
img_bytes = buf.getvalue()

res = preprocess_image(img_bytes)
res.pil_image.save("scratch/yellow_sticky_preprocessed.png")

print("Is digital canvas threshold triggering?")
bgr = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print("Gray > 240 ratio:", (gray > 240).sum() / gray.size)

recognizer = get_recognizer()
rec = recognizer.recognize(res.pil_image, num_samples=3)
print("Recognized LaTeX:", rec.latex)
