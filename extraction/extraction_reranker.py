from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-Reranker-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

onnx_model = ORTModelForSequenceClassification.from_pretrained(
    model_id,
    export=True,
    provider="CPUExecutionProvider"
)

onnx_model.save_pretrained("./model_repository/qwen3_reranker/1")
tokenizer.save_pretrained("./model_repository/qwen3_reranker/1")

print("✅ Qwen3-Reranker успешно экспортирована в ONNX!")