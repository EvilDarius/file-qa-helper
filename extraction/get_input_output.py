import onnx
from onnx import version_converter

model = onnx.load("../model_repository/m3-embeddings/1/model.onnx")

"""
print("🔹 Inputs:")
for inp in model.graph.input:
    dims = [d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim]
    print(f"  - {inp.name}: {dims}")

print("\n🔹 Outputs:")
for out in model.graph.output:
    dims = [d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim]
    print(f"  - {out.name}: {dims}")
"""

converted_model = version_converter.convert_version(model, 5)