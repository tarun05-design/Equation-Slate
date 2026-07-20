import sys
from pathlib import Path
from PIL import Image

img = Image.open("scratch/synthetic_handwritten_test.png")

try:
    import torch
    from transformers import AutoProcessor, VisionEncoderDecoderModel

    print("Loading TrOCR model...")
    processor = AutoProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten")
    
    # Avoid meta device initialization in transformers
    model = VisionEncoderDecoderModel.from_pretrained(
        "fhswf/TrOCR_Math_handwritten",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    model.to("cpu")
    model.eval()

    inputs = processor(img.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(inputs.pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("SUCCESS! TrOCR Output:", generated_text)
except Exception as exc:
    import traceback
    traceback.print_exc()
