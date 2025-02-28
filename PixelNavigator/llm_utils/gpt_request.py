import os
# import replicate
import openai
import base64
import cv2
import numpy as np
import os
from mimetypes import guess_type

dashscope_key = os.environ["DASHSCOPE_API_KEY"]
print(f'use dashscope_key {dashscope_key}')
dashscope_url = os.environ["DASHSCOPE_URL"]
print(f'use dashscope_url {dashscope_url}')
dashscope_model = os.environ["DASHSCOPE_MODEL"]
print(f'use dashscope_model {dashscope_model}')
# client = openai.OpenAI(api_key=openai_key)
gpt4v_client = openai.OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=dashscope_url,
)
# gpt4_client = AzureOpenAI(
#     api_key=gpt4_api_key,
#     api_version=api_version,
#     base_url=f"{gpt4_api_base}/openai/deployments/{deployment_name}"
# )
#
# gpt4v_api_base = os.environ['OPENAI_API_ENDPOINT']
# gpt4v_api_key= os.environ['OPENAI_API_KEY']
# deployment_name = 'gpt-4-vision-preview'
# api_version = '2024-02-15-preview'
# gpt4v_client = AzureOpenAI(
#     api_key=gpt4v_api_key,
#     api_version=api_version,
#     base_url=f"{gpt4v_api_base}/openai/deployments/{deployment_name}")

# Function to encode a local image into data URL 
def local_image_to_data_url(image):
    if isinstance(image,str):
        mime_type, _ = guess_type(image)
        with open(image, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    elif isinstance(image,np.ndarray):
        base64_encoded_data = base64.b64encode(cv2.imencode('.jpg',image)[1]).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_encoded_data}"

def gptv_response(text_prompt,image_prompt,system_prompt=""):
    prompt = [{'role':'system','content':system_prompt},
             {'role':'user','content':[{'type':'text','text':text_prompt},
                                       {'type':'image_url','image_url':{'url':local_image_to_data_url(image_prompt)}}]}]
    response = gpt4v_client.chat.completions.create(model=dashscope_model,
                                                    messages=prompt,
                                                    max_tokens=1000)
    return response.choices[0].message.content

def gpt_response(text_prompt,system_prompt=""):
    prompt = [{'role':'system','content':system_prompt},
              {'role':'user','content':[{'type':'text','text':text_prompt}]}]
    response = gpt4v_client.chat.completions.create(model=dashscope_model,
                                              messages=prompt,
                                              max_tokens=1000)
    return response.choices[0].message.content

