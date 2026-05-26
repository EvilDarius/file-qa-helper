import numpy as np
import json
import triton_python_backend_utils as pb_utils
from typing import List, Dict
from collections import defaultdict
from transformers import AutoTokenizer
from scipy.sparse import csr_matrix
import onnxruntime as ort


class TritonPythonModel:
    """
    Triton Python Backend для ONNX модели эмбеддингов
    """

    def initialize(self, args):
        """
        Инициализация модели при загрузке
        
        Parameters в model_config должны включать:
        - model_path: путь к ONNX модели
        - tokenizer_path: путь к токенизатору
        - max_length: максимальная длина последовательности (default: 8192)
        - use_fp16: использовать ли float16 (default: true)
        """
        self.model_config = json.loads(args['model_config'])
        
        # Получаем параметры из конфигурации
        params = self.model_config.get('parameters', {})
        
        model_path = params.get('model_path', {}).get('string_value', '/models/m3-embeddings/1/model.onnx')
        tokenizer_path = params.get('tokenizer_path', {}).get('string_value', '/models/m3-embeddings/1/tokenizer')
        self.max_length = int(params.get('max_length', {}).get('string_value', '8192'))
        self.use_fp16 = params.get('use_fp16', {}).get('string_value', 'true').lower() == 'true'
        self.batch_size = int(params.get('internal_batch_size', {}).get('string_value', '12'))
        
        # Настройка ONNX Runtime
        providers = [('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 5 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        })]
        
        so = ort.SessionOptions()
        so.enable_mem_pattern = True
        so.enable_mem_reuse = True
        so.add_session_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0;gpu:0")
        so.add_session_config_entry('session.use_device_allocator_for_initializers', "1")
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            # Попытка загрузить с GPU
            self.session = ort.InferenceSession(
                model_path, 
                providers=providers, 
                sess_options=so
            )
        except Exception as e:
            print(f"Warning: Failed to load with GPU providers: {e}")
            print("Attempting to load with CPU provider...")
            # Fallback на CPU если есть проблемы с opset
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider'],
                sess_options=so
            )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        
        # Определяем специальные токены
        self.unused_tokens = set([
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        ])

    def _process_token_weights(self, token_weights: np.ndarray, input_ids: list) -> Dict:
        """
        Обработка весов токенов для sparse embeddings
        Возвращает словарь вместо csr_matrix для сериализации
        """
        result = defaultdict(float)
        
        for w, idx in zip(token_weights, input_ids):
            if idx not in self.unused_tokens and w > 0:
                result[idx] = max(result[idx], float(w))
        
        indices = list(result.keys())
        data = list(result.values())
        
        return {
            'indices': indices,
            'data': data,
            'vocab_size': self.tokenizer.vocab_size
        }

    def execute(self, requests):
        """
        Обработка batch запросов
        
        Ожидаемые входы:
        - INPUT_TEXT: массив строк (TYPE_STRING)
        - RETURN_DENSE: bool (TYPE_BOOL) - возвращать ли dense embeddings
        - RETURN_SPARSE: bool (TYPE_BOOL) - возвращать ли sparse embeddings
        
        Выходы:
        - DENSE_EMBEDDINGS: массив dense векторов (TYPE_FP32 или TYPE_FP16)
        - SPARSE_EMBEDDINGS: JSON строка со sparse данными (TYPE_STRING)
        """
        responses = []
        
        for request in requests:
            # Получаем входные данные
            input_text = pb_utils.get_input_tensor_by_name(request, "TEXT")
            return_dense_tensor = pb_utils.get_input_tensor_by_name(request, "RETURN_DENSE")
            return_sparse_tensor = pb_utils.get_input_tensor_by_name(request, "RETURN_SPARSE")
            
            # Декодируем текст
            sentences = [s.decode('utf-8') if isinstance(s, bytes) else s 
                        for s in input_text.as_numpy().flatten()]
            
            # Получаем флаги
            return_dense = return_dense_tensor.as_numpy()[0] if return_dense_tensor else True
            return_sparse = True
            
            # Обработка
            dense_embeddings = []
            sparse_embeddings = []
            
            # Обрабатываем батчами
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i:i+self.batch_size]
                
                # Токенизация
                inputs = self.tokenizer(
                    batch,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Подготовка входов для ONNX
                ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
                
                # Inference
                ort_outputs = self.session.run(None, ort_inputs)
                
                # Обработка dense embeddings
                if return_dense:
                    batch_dense = ort_outputs[0]
                    if self.use_fp16:
                        batch_dense = batch_dense.astype(np.float16)
                    dense_embeddings.extend(batch_dense)
                
                # Обработка sparse embeddings
                if return_sparse:
                    sparse_vecs = ort_outputs[1]
                    for j, input_ids in enumerate(inputs["input_ids"].cpu().numpy()):
                        sparse_embeddings.append(
                            self._process_token_weights(sparse_vecs[j], input_ids.tolist())
                        )
            
            # Формируем выходы
            output_tensors = []
            
            if return_dense:
                dense_array = np.array(dense_embeddings)
                dense_tensor = pb_utils.Tensor("DENSE", dense_array)
                output_tensors.append(dense_tensor)
            
            if return_sparse:
                # Сериализуем sparse данные в JSON
                sparse_json = json.dumps(sparse_embeddings)
                sparse_array = np.array([sparse_json.encode('utf-8')], dtype=object)
                sparse_tensor = pb_utils.Tensor("SPARSE", sparse_array)
                output_tensors.append(sparse_tensor)
            
            # Создаем ответ
            inference_response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        """Очистка ресурсов"""
        print('Cleaning up model resources...')