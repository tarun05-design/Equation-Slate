import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model_inference import get_recognizer
from core.equation_solver import solve_expression

img = Image.open("scratch/latest_preprocessed.png")
recognizer = get_recognizer()

res = recognizer.recognize(img, num_samples=5)
print("Decoded samples:")
for i, sample in enumerate(res.samples):
    print(f"  Sample {i+1}: {sample}")

print("\nBest LaTeX:", res.latex)

try:
    solved = solve_expression(res.latex)
    print("Solved:", solved)
except Exception as e:
    print("Solver error:", e)
