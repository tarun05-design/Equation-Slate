import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Create synthetic handwritten image: (x + 3)^4 + (x + 1)^4 = 82
img = Image.new("RGB", (650, 200), "white")
draw = ImageDraw.Draw(img)

try:
    font_main = ImageFont.truetype("arial.ttf", 48)
    font_sub = ImageFont.truetype("arial.ttf", 32)
except OSError:
    font_main = font_sub = ImageFont.load_default()

draw.text((30, 70), "(x + 3)", fill="black", font=font_main)
draw.text((175, 40), "4", fill="black", font=font_sub)
draw.text((205, 70), "+ (x + 1)", fill="black", font=font_main)
draw.text((385, 40), "4", fill="black", font=font_sub)
draw.text((415, 70), "= 82", fill="black", font=font_main)

img.save("scratch/synthetic_handwritten_test.png")

print("Synthetic image created.")
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    print("Loading TrOCR_Math_handwritten...")
    processor = TrOCRProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")
    
    pixel_values = processor(img.convert("RGB"), return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("TrOCR Recognized LaTeX:", generated_text)
except Exception as exc:
    print("TrOCR test error:", exc)
