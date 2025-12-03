import os
from openai import OpenAI

from dotenv import load_dotenv  

load_dotenv(override=True)

openai=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model_1="gpt-4o-mini"
model_2="gpt-4o-mini"

messages_1=[{"role":"user","content":"Give me a question asking capital city of any country"}]

def get_model_response(model, messages_1):
    response = openai.chat.completions.create(model=model, messages=messages_1)
    return response.choices[0].message.content

def get_model_response_with_chaining(model, messages_2):
    response = openai.chat.completions.create(model=model, messages=messages_2)
    return response.choices[0].message.content


response_1 = get_model_response(model_1, messages_1)
print(response_1)

messages_2 = [{"role":"user","content":response_1}]
response_2 = get_model_response_with_chaining  (model_2, messages_2)
print(response_2)




