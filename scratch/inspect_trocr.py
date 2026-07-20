import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")

print("Decoder type:", type(model.decoder))
print("Decoder model type:", type(model.decoder.model))
print("Decoder model decoder type:", type(model.decoder.model.decoder))
embed_pos = model.decoder.model.decoder.embed_positions
print("embed_positions type:", type(embed_pos))
print("embed_positions dir:", dir(embed_pos))
print("_float_tensor:", getattr(embed_pos, "_float_tensor", None))
