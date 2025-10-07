from enum import Enum


class ClassificationModel(str, Enum):
    # llama3 = "llama3.3:70b-instruct-q3_K_M"
    llama_small = "llama3.2:3b"
    owned_mistral = "my-own-model:latest"
    qwen = "qwen3:8b"
    deepseek = "deepseek-r1:8b"
    supernove = "Supernova-Medius"
