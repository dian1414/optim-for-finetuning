################## modelscope ####################
import time
from modelscope.hub.snapshot_download import snapshot_download
from modelscope import HubApi
api=HubApi()
api.login('5d088a96-4dcf-4da1-849a-ee71a3a2dea8')


repo_id = "LLM-Research/Llama3-ChatQA-1.5-8B"
cache_dir = './meta-llama/Llama3-ChatQA-1.5-8B'
time_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(time_start)
while True:
    try:
        snapshot_download(
            cache_dir=cache_dir,
            model_id=repo_id,
            revision='master'
        )
        print('finish downloading.')
        break
    except Exception as e :
        print(e)
time_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f"start time: {time_start}")
print(f"end time: {time_end}")
