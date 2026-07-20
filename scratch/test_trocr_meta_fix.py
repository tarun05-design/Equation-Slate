import sys
from pathlib import Path
from PIL import Image

img = Image.open("scratch/synthetic_handwritten_test.png")

try:
    import torch
    from transformers import AutoProcessor, VisionEncoderDecoderModel

    print("Loading TrOCR model with meta tensor fix...")
    processor = AutoProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")
    
    # Fix PyTorch 2.3+ meta tensor bug in TrOCR embed_positions
    if hasattr(model.decoder.model.decoder.embed_positions, "_float_tensor"):
        if model.decoder.model.decoder.embed_positions._float_tensor.is_meta:
            model.decoder.model.decoder.embed_positions._float_tensor = torch.zeros(1)
            
    model.eval()

    inputs = processor(img.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(inputs.pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("SUCCESS! TrOCR Output:", generated_text)
except Exception as exc:
    import traceback
    traceback.print_exc()
