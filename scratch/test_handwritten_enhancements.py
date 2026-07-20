import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import normalize_polarity, crop_to_content
from core.model_inference import get_recognizer

def process_handwritten_photo(gray: np.ndarray) -> np.ndarray:
    # 1. Mild bilateral filter or Gaussian blur to smooth paper grain without blurring stroke edges
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Otsu thresholding to separate ink from paper background cleanly
    _, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Stroke enhancement: dilate slightly if strokes are too thin (ink is 0/black)
    # Invert binary so ink is white (255) for morphological operations
    ink_white = cv2.bitwise_not(binary)
    
    # Kernel for slight stroke thickening (preserves handwritten structure while giving printed-like stroke weight)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thickened_ink = cv2.dilate(ink_white, kernel, iterations=1)
    
    # Invert back to black ink on white background
    enhanced_binary = cv2.bitwise_not(thickened_ink)
    
    # 4. Crop with comfortable padding for exponents/subscripts
    cropped = crop_to_content(enhanced_binary, padding=30)
    return cropped

# Create a test synthetic handwritten-style image
img = Image.new("RGB", (700, 250), (235, 238, 242)) # real paper tint
draw = ImageDraw.Draw(img)

# Add shadow gradient
np_img = np.array(img).astype(float)
for r in range(250):
    np_img[r, :, :] -= (r * 0.1)
np_img = np.clip(np_img, 0, 255).astype(np.uint8)
img = Image.fromarray(np_img)
draw = ImageDraw.Draw(img)

try:
    font_large = ImageFont.truetype("arial.ttf", 64)
    font_small = ImageFont.truetype("arial.ttf", 44)
except OSError:
    font_large = font_small = ImageFont.load_default()

# Render 4^x + 2^x = 20
draw.text((60, 90), "4", fill=(40, 42, 50), font=font_large)
draw.text((105, 55), "x", fill=(40, 42, 50), font=font_small)
draw.text((170, 90), "+", fill=(40, 42, 50), font=font_large)
draw.text((250, 90), "2", fill=(40, 42, 50), font=font_large)
draw.text((295, 55), "x", fill=(40, 42, 50), font=font_small)
draw.text((350, 90), "= 20", fill=(40, 42, 50), font=font_large)

import io
buf = io.BytesIO()
img.save(buf, format="PNG")
raw_bytes = buf.getvalue()

bgr = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
gray = normalize_polarity(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

processed = process_handwritten_photo(gray)
pil_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB))
pil_img.save("scratch/test_handwritten_processed.png")

rec = get_recognizer()
res = rec.recognize(pil_img, num_samples=1)
print("Handwritten photo result:", res.latex)
