import cv2
import numpy as np
from PIL import Image, ImageDraw

# 1. Digital canvas export
canvas_img = Image.new("RGB", (800, 360), "#FFFFFF")
draw = ImageDraw.Draw(canvas_img)
draw.text((100, 100), "2x - 3 = -7", fill="#000000")

import io
buf1 = io.BytesIO()
canvas_img.save(buf1, format="PNG")

bgr1 = cv2.imdecode(np.frombuffer(buf1.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
gray1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
mid_range_1 = ((gray1 > 30) & (gray1 < 230)).sum() / gray1.size
print("Digital Canvas mid-range pixel ratio (30-230):", mid_range_1)

# 2. Yellow sticky note photo simulation
sticky_img = Image.new("RGB", (800, 400), "white")
draw2 = ImageDraw.Draw(sticky_img)
draw2.rectangle([100, 50, 700, 350], fill="#F9E79F") # yellow
draw2.text((150, 100), "2y - 3x = -4", fill="red")

buf2 = io.BytesIO()
sticky_img.save(buf2, format="PNG")

bgr2 = cv2.imdecode(np.frombuffer(buf2.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
mid_range_2 = ((gray2 > 30) & (gray2 < 230)).sum() / gray2.size
print("Sticky Note mid-range pixel ratio (30-230):", mid_range_2)
