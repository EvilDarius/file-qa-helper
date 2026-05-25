from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoModel, AutoTokenizer

model_id = "BAAI/bge-m3"

# Сохраняем токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Экспорт модели в ONNX
onnx_model = AutoModel.from_pretrained(
    model_id,
)

onnx_model.save_pretrained("./model_repository/bge-m3/1")
tokenizer.save_pretrained("./model_repository/bge-m3/1")

print("✅ BAAI/bge-m3 успешно экспортирована в ONNX!")