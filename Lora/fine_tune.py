from datasets import Dataset
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
train_list=[]
val_list=[]
with open('./dataset/output_train.json', 'r', encoding='utf-8') as file:  
    data = json.load(file)

for i in range(0,len(data)):
    train_list.append({'text':'Instruction:'+data[i]['instruction']+'\nOutput:'+data[i]['output']+'</s>'})

# for i in range(150,635):
#     train_list.append({'text':'Instruction:'+data[i]['instruction']+'\nInput:'+data[i]['input']+'\nOutput:'+data[i]['output']+'</s>'})

train_list=Dataset.from_dict({key:[dic[key] for dic in train_list] for key in train_list[0]})
# val_list=Dataset.from_dict({key:[dic[key] for dic in val_list] for key in val_list[0]})

# fine-tune

import torch
from peft import LoraConfig,get_peft_model
from transformers import TrainingArguments,AutoModelForCausalLM,AutoTokenizer

peft_config=LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0,
    bias='none',
    task_type="CAUSAL_LM"
)
output_dir='./model_chatqa_normal'
training_args=TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    optim='adamw_torch',
    learning_rate=2e-5,
    eval_steps=50,
    save_steps=98,
    logging_steps=1,
    evaluation_strategy="no",
    group_by_length=False,
    num_train_epochs=5,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    bf16=True,
    lr_scheduler_type='cosine',
    warmup_steps=20
)

model_path="./meta-llama/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct"
model=AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto'
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.enable_input_require_grads()
model=get_peft_model(model,peft_config)
model.print_trainable_parameters()
model.config.use_cache=False

# model = model.to('cpu')
tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
tokenizer.pad_token_id=0
tokenizer.padding_side='right'
print(val_list)
# train
from trl import SFTTrainer
trainer=SFTTrainer(
    model=model,
    train_dataset=train_list,
    eval_dataset=None,
    dataset_text_field='text',
    peft_config=peft_config,
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_args
)

trainer.train()
trainer.model.save_pretrained(output_dir)
