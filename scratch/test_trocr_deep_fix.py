import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")

# Deep fix for _float_tensor in any submodule or child object
def fix_all_meta_tensors(root):
    visited = set()
    def _recurse(obj):
        if id(obj) in visited:
            return
        visited.add(id(obj))
        
        if hasattr(obj, "_float_tensor"):
            tensor = getattr(obj, "_float_tensor")
            if isinstance(tensor, torch.Tensor) and tensor.is_meta:
                real_tensor = torch.zeros(1, dtype=torch.float32, device="cpu")
                setattr(obj, "_float_tensor", real_tensor)
                if hasattr(obj, "_buffers"):
                    obj._buffers["_float_tensor"] = real_tensor

        if isinstance(obj, torch.nn.Module):
            for child in obj.children():
                _recurse(child)
            for _, child in obj._modules.items():
                if child is not None:
                    _recurse(child)

fix_all_meta_tensors(model)
print("Deep meta fix applied!")

img = Image.open("scratch/synthetic_handwritten_test.png")
processor = AutoProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten")
inputs = processor(img.convert("RGB"), return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(inputs.pixel_values)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("SUCCESS! TrOCR Output:", output)
