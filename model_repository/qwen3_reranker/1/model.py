"""
Triton Python Backend для модели qwen3-reranker
Структура директорий:
model_repository/
└── qwen3_reranker/
    ├── config.pbtxt
    ├── 1/
    │   └── model.py
"""

# ============================================
# config.pbtxt
# ============================================
"""
name: "qwen3_reranker"
backend: "python"
max_batch_size: 8

input [
  {
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [-1]
  }
]

output [
  {
    name: "SCORES"
    data_type: TYPE_FP32
    dims: [-1]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 100
}
"""

# ============================================
# model.py (сохранить в директорию 1/)
# ============================================

import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TritonPythonModel:
    """Triton Python Backend для Qwen3-Reranker модели"""

    def initialize(self, args):
        """
        Инициализация модели при запуске Triton сервера
        
        Args:
            args: dict с параметрами модели
        """
        self.model_config = json.loads(args['model_config'])
        
        # Получаем параметры из конфигурации
        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "logits"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type']
        )
        
        # Определяем устройство
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Путь к модели (можно передать через environment variables)
        model_path = "/models/qwen3_reranker/1"
        # Загружаем токенайзер и модель
        self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            local_files_only=True
        ).to(self.device)
        self.model.eval()
        
        print(f"Модель Qwen3-Reranker загружена на {self.device}")

    def execute(self, requests):
        """
        Обработка batch запросов
        
        Args:
            requests: список InferenceRequest объектов
            
        Returns:
            список InferenceResponse объектов
        """
        responses = []
        
        for request in requests:
            # Получаем входные данные
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            texts = input_tensor.as_numpy()
            
            # Декодируем строки из bytes
            texts = [t.decode('utf-8') if isinstance(t, bytes) else t for t in texts.flatten()]
            
            # Обрабатываем тексты
            # Формат для reranker: ["query", "document1", "document2", ...]
            # Первый элемент - query, остальные - документы для ранжирования
            if len(texts) < 2:
                # Если меньше 2 элементов, возвращаем ошибку
                error = pb_utils.TritonError(
                    "Требуется минимум 2 текста: query и хотя бы один документ"
                )
                responses.append(pb_utils.InferenceResponse(error=error))
                continue
            
            query = texts[0]
            documents = texts[1:]
            
            # Создаем пары query-document
            pairs = [[query, doc] for doc in documents]
            
            try:
                with torch.no_grad():
                    # Токенизация
                    inputs = self.tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Получаем scores
                    outputs = self.model(**inputs)
                    scores = outputs.logits.squeeze(-1).cpu().numpy()
                    
                    # Применяем softmax для нормализации (опционально)
                    # scores = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                
                # Создаем выходной тензор
                scores = scores.astype(self.output_dtype)
                output_tensor = pb_utils.Tensor("logits", scores)
                
                # Создаем ответ
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(inference_response)
                
            except Exception as e:
                error = pb_utils.TritonError(f"Ошибка при обработке: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))
        
        return responses

    def finalize(self):
        """Освобождение ресурсов при остановке сервера"""
        print("Очистка ресурсов модели Qwen3-Reranker")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
