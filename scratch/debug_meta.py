import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")

# Fix PyTorch 2.3+ meta tensor bug on TrOCR weights
for name, module in model.named_modules():
    if hasattr(module, "_float_tensor") and getattr(module, "_float_tensor").is_meta:
        module._float_tensor = torch.zeros(1, dtype=torch.float32)

img = Image.open("scratch/synthetic_handwritten_test.png")
processor = AutoProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten")
inputs = processor(img.convert("RGB"), return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(inputs.pixel_values)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("SUCCESS! TrOCR Output:", output)
