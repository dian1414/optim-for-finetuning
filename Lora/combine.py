from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import os 

base_model_path="./meta-llama/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct"
finetune_model_path="./model"

tokenizer=AutoTokenizer.from_pretrained(base_model_path,trust_remote_code=True)
model=AutoPeftModelForCausalLM.from_pretrained(finetune_model_path,device_map='auto',torch_dtype=torch.bfloat16)

model=model.merge_and_unload()
model.save_pretrained('./llama3-instruct-8b-finetuned')
