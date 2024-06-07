from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
with open('./dataset/output_test.json','r') as f:
    data=json.load(f)
model_id = "./meta-llama/Llama3-ChatQA-1.5-8B/LLM-Research/Llama3-ChatQA-1___5-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")


def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation
    
    return formatted_input
ans=[]
with tqdm(total=len(data)) as pbar:
    for i in range(0,len(data)):
        messages = [
            {"role": "user", "content": data[i]['input']}
        ]
        document=data[i]['instruction']
        formatted_input = get_formatted_input(messages, document)
        tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, eos_token_id=terminators)

        response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
        ans.append(tokenizer.decode(response, skip_special_tokens=True))
        pbar.update(1)

# print(tokenizer.decode(response, skip_special_tokens=True))
with open('chatqa_512.json', 'w') as file:
    json.dump(ans, file, indent=4)