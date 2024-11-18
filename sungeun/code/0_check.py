from transformers import AutoModel

model = AutoModel.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
print(model)