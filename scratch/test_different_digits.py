import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.preprocessing import preprocess_image
from core.model_inference import get_recognizer

recognizer = get_recognizer()

def test_drawing(draw_fn, title):
    img = Image.new("RGB", (800, 360), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw_fn(draw)
    
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    res = preprocess_image(buf.getvalue())
    rec = recognizer.recognize(res.pil_image, num_samples=3)
    print(f"[{title}] -> {rec.latex}")

# 1. Standard 2 (flat bottom) + 2
def draw_flat_2(draw):
    # 2: curve top, diagonal down, flat bottom
    draw.line([(100, 120), (120, 100), (140, 100), (150, 120), (100, 180), (160, 180)], fill="#000000", width=8)
    # +
    draw.line([(220, 140), (280, 140)], fill="#000000", width=8)
    draw.line([(250, 110), (250, 170)], fill="#000000", width=8)
    # 2: curve top, diagonal down, flat bottom
    draw.line([(340, 120), (360, 100), (380, 100), (390, 120), (340, 180), (400, 180)], fill="#000000", width=8)

test_drawing(draw_flat_2, "Standard Flat-Bottom 2 + 2")

# 2. x^2 + 3x - 4 = 0
def draw_quadratic(draw):
    # x
    draw.line([(100, 130), (140, 180)], fill="#000000", width=8)
    draw.line([(140, 130), (100, 180)], fill="#000000", width=8)
    # ^2
    draw.line([(150, 110), (160, 100), (170, 100), (150, 130), (170, 130)], fill="#000000", width=6)
    # + 3x - 4 = 0
    draw.line([(190, 150), (230, 150)], fill="#000000", width=8)
    draw.line([(210, 130), (210, 170)], fill="#000000", width=8)

test_drawing(draw_quadratic, "x^2 + ...")
