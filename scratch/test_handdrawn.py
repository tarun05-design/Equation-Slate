import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import preprocess_image, denoise, threshold, crop_to_content, resize_with_padding
from core.model_inference import get_recognizer

# Create canvas image with hand-drawn style 2 + 2
img = Image.new("RGB", (800, 360), "#FFFFFC")
draw = ImageDraw.Draw(img)

# Draw "2"
points_2_1 = [(100, 100), (120, 80), (140, 90), (140, 120), (100, 180), (150, 180)]
draw.line(points_2_1, fill="#23283A", width=6, joint="round")

# Draw "+"
draw.line([(220, 130), (280, 130)], fill="#23283A", width=6)
draw.line([(250, 100), (250, 160)], fill="#23283A", width=6)

# Draw "2"
points_2_2 = [(350, 100), (370, 80), (390, 90), (390, 120), (350, 180), (400, 180)]
draw.line(points_2_2, fill="#23283A", width=6, joint="round")

import io
buf = io.BytesIO()
img.save(buf, format="PNG")
img_bytes = buf.getvalue()

res = preprocess_image(img_bytes)

# Save intermediate debug stages to inspect
cv2.imwrite("scratch/stage_gray.png", res.debug_stages["grayscale"])
cv2.imwrite("scratch/stage_denoised.png", res.debug_stages["denoised"])
cv2.imwrite("scratch/stage_binary.png", res.debug_stages["binary"])
cv2.imwrite("scratch/stage_cropped.png", res.debug_stages["cropped"])
cv2.imwrite("scratch/stage_resized.png", res.debug_stages["resized"])

recognizer = get_recognizer()
recognition = recognizer.recognize(res.pil_image, num_samples=3)
print("Recognized LaTeX:", recognition.latex)
print("Confidence:", recognition.confidence)
print("Samples:", recognition.samples)
