import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import preprocess_image, crop_to_content
from core.model_inference import get_recognizer

# Recreate user's image: 800x400 image, 75% pure white (#FFFFFF), center yellow sticky note (#F7E398)
img = Image.new("RGB", (800, 400), "#FFFFFF")
draw = ImageDraw.Draw(img)

# Yellow sticky note in center (300x200)
draw.rectangle([250, 100, 550, 300], fill="#F7E398")

# Draw "2y - 3x = -4" inside sticky note
try:
    font = ImageFont.truetype("arial.ttf", 60)
except Exception:
    font = ImageFont.load_default()

draw.text((260, 150), "2y", fill="black", font=font)
draw.text((320, 150), "-3x", fill="red", font=font)
draw.text((420, 150), "=-4", fill="black", font=font)

import io
buf = io.BytesIO()
img.save(buf, format="PNG")
img_bytes = buf.getvalue()

bgr = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
ratio_240 = (gray > 240).sum() / gray.size
print("Gray > 240 ratio:", ratio_240)

# Run current preprocessing
res = preprocess_image(img_bytes)
res.pil_image.save("scratch/sticky_issue_preprocessed.png")

# Debug thresholding:
_, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
cv2.imwrite("scratch/sticky_binary_issue.png", binary)

recognizer = get_recognizer()
rec = recognizer.recognize(res.pil_image, num_samples=3)
print("Recognized LaTeX:", rec.latex)
