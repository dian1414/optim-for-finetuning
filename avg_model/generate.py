import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM,PreTrainedTokenizerFast,AutoModelForCausalLM
from peft import PeftModel


class CombinedLanguageModel(nn.Module):
    def __init__(self, base_model_path, finetuned_model_path1, finetuned_model_path2):
        super(CombinedLanguageModel, self).__init__()

        model=LlamaForCausalLM.from_pretrained(base_model_path,load_in_8bit=False,device_map='auto',torch_dtype=torch.bfloat16)
        self.model1 = PeftModel.from_pretrained(model,finetuned_model_path1,map='auto')
        self.model2 = PeftModel.from_pretrained(model,finetuned_model_path2,map='auto')
        
    def forward(self, input_ids, attention_mask=None):

        output1 = self.model1(input_ids, attention_mask=attention_mask).logits
        output2 = self.model2(input_ids, attention_mask=attention_mask).logits
        
        avg_output = (output1 + output2) / 2
        return avg_output

finetuned_model_path1 = './llama2_sft/ft_llama/model_chatqa_normal'  
finetuned_model_path2 = './LongLoRA/model_chatqa' 
base_model_path='/home/zhangyan/Dynathink/llama2_sft/ft_llama/meta-llama/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct'

combined_model = CombinedLanguageModel(base_model_path,finetuned_model_path1, finetuned_model_path2)

tokenizer=PreTrainedTokenizerFast.from_pretrained(base_model_path,trust_remote_code=True,map='auto')

model_save_path = 'combined_language_model.pth'

loaded_model = CombinedLanguageModel(base_model_path,finetuned_model_path1, finetuned_model_path2)
loaded_model.load_state_dict(torch.load(model_save_path))

import json
from tqdm import tqdm
with open('output_test.json') as f:
    data=json.load(f)
preprompt='Read the instructions below and answer the question(Only give out the output).'

res_list=[]
with tqdm(total=30) as pbar:
    for i in range(0,30):
        prompt='Instruction:'+data[i]["instruction"]+'\nOutput:'
        input=tokenizer(preprompt+prompt,return_tensors='pt').to('cuda')
        # model.eval()
        with torch.no_grad():
            ans = loaded_model(input.input_ids, attention_mask=input.attention_mask)
            probs = torch.nn.functional.softmax(ans, dim=-1)
            predicted_tokens = torch.argmax(probs, dim=-1)
            result = tokenizer.decode(predicted_tokens[0],skip_special_tokens=True)
            # print(result)
            # result=model.generate(**input,max_new_tokens=256)[0]
            # ans=tokenizer.decode(result,skip_special_tokens=True)
            # if ans.find('Output')!=-1:
            #     ans=ans[ans.find('Output'):len(ans)]
            #     ans=ans[ans.find('Output'):ans.find('[')]
            #     ans=ans[7:len(ans)]
            pair={}
            pair['instruction']=data[i]['instruction']
            pair['gold_answer']=result
        res_list.append(pair)
        pbar.update(1)
# res_list=res_list.tolist()
with open('result_chatqa_combined1.json', 'w') as file:
    json.dump(res_list, file, indent=4)