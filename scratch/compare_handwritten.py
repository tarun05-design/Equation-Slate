import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model_inference import get_recognizer
from core.preprocessing import preprocess_image

# Render handwritten-style image: (x+3)^4 + (x+1)^4 = 82
img = Image.new("RGB", (700, 220), (242, 244, 248))
draw = ImageDraw.Draw(img)

try:
    font_main = ImageFont.truetype("arial.ttf", 52)
    font_sub = ImageFont.truetype("arial.ttf", 36)
except OSError:
    font_main = font_sub = ImageFont.load_default()

draw.text((40, 75), "(x + 3)", fill=(35, 38, 45), font=font_main)
draw.text((200, 42), "4", fill=(35, 38, 45), font=font_sub)
draw.text((235, 75), "+ (x + 1)", fill=(35, 38, 45), font=font_main)
draw.text((435, 42), "4", fill=(35, 38, 45), font=font_sub)
draw.text((470, 75), "= 82", fill=(35, 38, 45), font=font_main)

import io
buf = io.BytesIO()
img.save(buf, format="PNG")
raw_bytes = buf.getvalue()

# 1. Existing Pix2Tex
res_prep = preprocess_image(raw_bytes)
rec_pix = get_recognizer()
pix_result = rec_pix.recognize(res_prep.pil_image, num_samples=1)
print("Pix2Tex Output:", pix_result.latex)

# 2. TrOCR Math Handwritten
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")
    
    pixel_values = processor(res_prep.pil_image.convert("RGB"), return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    trocr_result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("TrOCR Math Output:", trocr_result)
except Exception as exc:
    print("TrOCR Error:", exc)
