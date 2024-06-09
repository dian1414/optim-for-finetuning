import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM,PreTrainedTokenizerFast,AutoModelForCausalLM
from peft import PeftModel


class CombinedLanguageModel(nn.Module):
    def __init__(self, base_model_path, finetuned_model_path1, finetuned_model_path2):
        super(CombinedLanguageModel, self).__init__()

        model=LlamaForCausalLM.from_pretrained(base_model_path,load_in_8bit=False,device_map='auto',torch_dtype=torch.bfloat16)
        self.model1 = PeftModel.from_pretrained(model,finetuned_model_path1)
        self.model2 = PeftModel.from_pretrained(model,finetuned_model_path2)
        
    def forward(self, input_ids, attention_mask=None):

        output1 = self.model1(input_ids, attention_mask=attention_mask).logits
        output2 = self.model2(input_ids, attention_mask=attention_mask).logits
        
        avg_output = (output1 + output2) / 2
        return avg_output

finetuned_model_path1 = './llama2_sft/ft_llama/model_chatqa_normal'  
finetuned_model_path2 = './LongLoRA/model_chatqa' 
base_model_path='/home/zhangyan/Dynathink/llama2_sft/ft_llama/meta-llama/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct'

combined_model = CombinedLanguageModel(base_model_path,finetuned_model_path1, finetuned_model_path2)

tokenizer=PreTrainedTokenizerFast.from_pretrained(base_model_path,trust_remote_code=True)

model_save_path = 'combined_language_model.pth'
torch.save(combined_model.state_dict(), model_save_path)