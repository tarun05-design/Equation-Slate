import io
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model_inference import get_recognizer
from core.preprocessing import preprocess_image

# Simulate canvas drawing: freehand (x+3)^4 + (x+1)^4 = 82 on 800x360 white canvas
canvas_img = Image.new("RGB", (800, 360), "#FFFFFF")
draw = ImageDraw.Draw(canvas_img)

# Freehand style line strokes
points = [
    # (x + 3)^4
    (50, 160), (70, 220), # (
    (110, 160), (150, 220), # x stroke 1
    (150, 160), (110, 220), # x stroke 2
    (180, 190), (220, 190), # + horizontal
    (200, 170), (200, 210), # + vertical
    (240, 165), (280, 165), (240, 195), (280, 195), (280, 225), (240, 225), # 3
    (300, 160), (320, 220), # )
    (330, 120), (330, 150), (360, 150), (360, 110), (360, 160), # 4 superscript
]

for i in range(0, len(points)-1, 2):
    draw.line([points[i], points[i+1]], fill="#000000", width=6, joint="round")

buf = io.BytesIO()
canvas_img.save(buf, format="PNG")
canvas_bytes = buf.getvalue()

prep_res = preprocess_image(canvas_bytes)
rec = get_recognizer()
result = rec.recognize(prep_res.pil_image, num_samples=1)

print("Canvas Recognition Result:", result.latex)
print("Confidence:", result.confidence)
