import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import preprocess_image
from core.model_inference import get_recognizer
import cv2

# Synthesize a canvas image similar to browser canvas (800x360, white background, #23283A stroke)
img = Image.new("RGB", (800, 360), "#FFFFFC")
draw = ImageDraw.Draw(img)

# Let's draw "2 + 2" using thick lines or font
try:
    font = ImageFont.truetype("arial.ttf", 100)
    draw.text((150, 100), "2 + 2", fill="#23283A", font=font)
except Exception:
    draw.text((150, 100), "2 + 2", fill="#23283A")

import io
buf = io.BytesIO()
img.save(buf, format="PNG")
img_bytes = buf.getvalue()

res = preprocess_image(img_bytes)
print("Quality score:", res.quality_score)
print("PIL image size:", res.pil_image.size)
res.pil_image.save("scratch/preprocessed_output.png")

recognizer = get_recognizer()
recognition = recognizer.recognize(res.pil_image, num_samples=1)
print("LaTeX output:", recognition.latex)
