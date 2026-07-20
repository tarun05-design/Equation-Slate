import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import resize_with_padding
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
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find connected components of ink (< 128)
ink_mask = (binary < 128).astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask)

# Filter out connected components that touch the image border or span > 80% width/height (card background borders)
H, W = binary.shape
valid_boxes = []
clean_binary = np.full_like(binary, 255)

for i in range(1, num_labels):
    x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
    # Check if component touches image border or is huge frame
    touches_border = (x == 0 or y == 0 or x + w == W or y + h == H)
    is_huge_frame = (w > 0.7 * W and h > 0.7 * H)
    
    if not is_huge_frame and area >= 5:
        # Keep this component
        clean_binary[labels == i] = 0
        valid_boxes.append((x, y, x + w, y + h))

# Check if clean_binary text is white-on-black inside card:
# If clean_binary is black background with white text, invert it!
if (clean_binary == 0).sum() / clean_binary.size > 0.3:
    clean_binary = cv2.bitwise_not(clean_binary)

# Crop around valid_boxes
if valid_boxes:
    x0 = min(b[0] for b in valid_boxes)
    y0 = min(b[1] for b in valid_boxes)
    x1 = max(b[2] for b in valid_boxes)
    y1 = max(b[3] for b in valid_boxes)
    p = 20
    y0 = max(0, y0 - p)
    x0 = max(0, x0 - p)
    y1 = min(H, y1 + p)
    x1 = min(W, x1 + p)
    cropped = clean_binary[y0:y1, x0:x1]
else:
    cropped = clean_binary

resized = resize_with_padding(cropped, 224, 224)
pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))
pil_img.save("scratch/dark_mode_no_frame.png")

recognizer = get_recognizer()
rec = recognizer.recognize(pil_img, num_samples=3)
print("Recognized LaTeX (no frame):", rec.latex)
