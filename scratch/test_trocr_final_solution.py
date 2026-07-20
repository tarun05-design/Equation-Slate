import math
import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")

def fix_trocr_meta_tensors(model):
    for m in model.modules():
        if hasattr(m, "weights") and getattr(m.weights, "is_meta", False):
            shape = m.weights.shape
            num_embeddings, embedding_dim = shape[0], shape[1]
            half_dim = embedding_dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
            emb = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, embedding_dim)
            if embedding_dim % 2 == 1:
                emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
            m.weights = torch.nn.Parameter(emb, requires_grad=False)

fix_trocr_meta_tensors(model)
model.eval()

img = Image.open("scratch/synthetic_handwritten_test.png")
processor = AutoProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten")
inputs = processor(img.convert("RGB"), return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(inputs.pixel_values)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("SUCCESS! TrOCR Output:", output)
