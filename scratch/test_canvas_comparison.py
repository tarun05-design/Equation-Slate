import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import preprocess_image, crop_to_content, resize_with_padding
from core.model_inference import get_recognizer

# Create canvas image with 5px strokes (similar to default pen in canvas)
img = Image.new("RGB", (800, 360), "#FFFFFC")
draw = ImageDraw.Draw(img)

# Draw "2 + 2" with 5px stroke width (matching screenshot)
# Left 2
draw.arc([100, 100, 160, 150], 180, 0, fill="#23283A", width=5)
draw.line([(160, 125), (100, 180)], fill="#23283A", width=5)
draw.line([(100, 180), (160, 180)], fill="#23283A", width=5)

# Plus
draw.line([(220, 140), (280, 140)], fill="#23283A", width=5)
draw.line([(250, 110), (250, 170)], fill="#23283A", width=5)

# Right 2
draw.arc([340, 100, 400, 150], 180, 0, fill="#23283A", width=5)
draw.line([(400, 125), (340, 180)], fill="#23283A", width=5)
draw.line([(340, 180), (400, 180)], fill="#23283A", width=5)

import io
buf = io.BytesIO()
img.save(buf, format="PNG")
img_bytes = buf.getvalue()

# Current preprocessing
res_current = preprocess_image(img_bytes)
res_current.pil_image.save("scratch/current_preprocessed.png")

# Improved preprocessing for clean digital/canvas images:
bgr = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Otsu or simple threshold directly on gray
_, binary_clean = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
cropped_clean = crop_to_content(binary_clean, padding=20)
resized_clean = resize_with_padding(cropped_clean, 224, 224)
pil_clean = Image.fromarray(cv2.cvtColor(resized_clean, cv2.COLOR_GRAY2RGB))
pil_clean.save("scratch/clean_preprocessed.png")

recognizer = get_recognizer()

rec_current = recognizer.recognize(res_current.pil_image, num_samples=3)
print("Current pipeline result:", rec_current.latex)

rec_clean = recognizer.recognize(pil_clean, num_samples=3)
print("Clean pipeline result:", rec_clean.latex)
