import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ['HF_TOKEN'] = "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"

tokenizer = AutoTokenizer.from_pretrained("CarrotAI/Llama-3.2-Rabbit-Ko-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("CarrotAI/Llama-3.2-Rabbit-Ko-3B-Instruct")

input_text = "너에 대해서 설명해 줄래?"
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_length=512)
print(tokenizer.decode(outputs[0]))

