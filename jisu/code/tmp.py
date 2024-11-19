# from transformers import AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("beomi/Qwen2.5-7B-Instruct-kowiki-qa-context")

print(model)
