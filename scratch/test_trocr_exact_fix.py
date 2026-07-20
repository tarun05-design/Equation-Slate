import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")

# Directly fix the meta buffer in TrOCRPositionalEmbedding
embed_pos_module = model.decoder.model.decoder.embed_positions
embed_pos_module._float_tensor = torch.tensor([0.0], dtype=torch.float32)

print("_float_tensor fixed. Is meta?", embed_pos_module._float_tensor.is_meta)

img = Image.open("scratch/synthetic_handwritten_test.png")
processor = AutoProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten")
inputs = processor(img.convert("RGB"), return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(inputs.pixel_values)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("SUCCESS! TrOCR Output:", output)
