import torch 
from transformers import LlamaTokenizer, LlamaForCausalLM,PreTrainedTokenizerFast,AutoModelForCausalLM
from peft import PeftModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
base_model_path="/home/zhangyan/Dynathink/llama2_sft/ft_llama/meta-llama/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct"
finetuned_model_path="./model_chatqa_moredata"
# merged_model_path="./llama3-instruct-8b-finetuned"

tokenizer=PreTrainedTokenizerFast.from_pretrained(base_model_path,trust_remote_code=True)

model=LlamaForCausalLM.from_pretrained(base_model_path,load_in_8bit=False,device_map='auto',torch_dtype=torch.bfloat16)
model=PeftModel.from_pretrained(model,finetuned_model_path)
# model=AutoModelForCausalLM.from_pretrained(merged_model_path,load_in_8bit=False,device_map='auto',torch_dtype=torch.bfloat16)

import json
from tqdm import tqdm
with open('./output_test.json') as f:
    data=json.load(f)
preprompt='Read the instructions below and answer the question(Only give out the output).'

res_list=[]
with tqdm(total=len(data)) as pbar:
    for i in range(0,len(data)):
        prompt='Instruction:'+data[i]["instruction"]+'\nOutput:'
        input=tokenizer(preprompt+prompt,return_tensors='pt').to('cuda')
        model.eval()
        with torch.no_grad():
            result=model.generate(**input,max_new_tokens=256)[0]
            ans=tokenizer.decode(result,skip_special_tokens=True)
            # if ans.find('Output')!=-1:
            #     ans=ans[ans.find('Output'):len(ans)]
            #     ans=ans[ans.find('Output'):ans.find('[')]
            #     ans=ans[7:len(ans)]
            pair={}
            pair['instruction']=data[i]['instruction']
            pair['gold_answer']=ans
        res_list.append(pair)
        pbar.update(1)

with open('result_chatqa_qlora_1.json', 'w') as file:
    json.dump(res_list, file, indent=4)