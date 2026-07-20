import torch
from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")

embed_pos = model.decoder.model.decoder.embed_positions
print("weights is_meta:", embed_pos.weights.is_meta)
print("weights shape:", embed_pos.weights.shape)

if embed_pos.weights.is_meta:
    # Initialize sinusoidal weights on CPU
    num_embeddings = embed_pos.weights.shape[0]
    embedding_dim = embed_pos.weights.shape[1]
    
    import math
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, embedding_dim)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        
    embed_pos.weights = torch.nn.Parameter(emb, requires_grad=False)

print("Fixed weights is_meta:", embed_pos.weights.is_meta)
