import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")

# Convert meta _float_tensor in TrOCRPositionalEmbedding modules
for m in model.modules():
    if hasattr(m, "_float_tensor"):
        m._float_tensor = torch.zeros(1, dtype=torch.float32)

print("Meta tensors fixed!")

img = Image.open("scratch/synthetic_handwritten_test.png")
processor = AutoProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten")
inputs = processor(img.convert("RGB"), return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(inputs.pixel_values)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("SUCCESS! TrOCR Output:", output)
