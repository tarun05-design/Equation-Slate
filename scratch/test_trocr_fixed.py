import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

img = Image.open("scratch/synthetic_handwritten_test.png")

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch

    print("Loading TrOCR_Math_handwritten with low_cpu_mem_usage=False...")
    processor = TrOCRProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(
        "fhswf/TrOCR_Math_handwritten", 
        low_cpu_mem_usage=False
    )
    model.eval()

    pixel_values = processor(img.convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("SUCCESS! TrOCR Recognized LaTeX:", generated_text)
except Exception as exc:
    print("TrOCR test error:", exc)
