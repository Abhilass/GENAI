from dotenv import load_dotenv
load_dotenv(override=True)

from codecs import utf_8_decode
import os
from dotenv.parser import Reader
from openai import OpenAI
import gradio as gr

print("vikas")
print(os.getenv("OPENAI_API_KEY"))

from pypdf import PdfReader

reader = PdfReader("/Users/apple/Abhilash_Projects/Abhilash_Sharma_15+yrs_Dev_DevOpsManager.pdf")
Resume=""
number_of_pages = len(reader.pages)

for i in range(number_of_pages):
    page = reader.pages[i]
    text = page.extract_text()
    if text:
     Resume += text
    print(Resume)

openai=OpenAI()

name="Abhilash Sharma"

with open("summary.txt",'r',encoding="utf-8") as f :
    summary=f.read()

print (summary)

system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer, say so."

system_prompt += f"\n\n## Summary:\n{summary}\n\n## Resume Profile:\n{Resume}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."


def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
    
    
    return response.choices[0].message.content

gr.ChatInterface(chat, type="messages").launch()


