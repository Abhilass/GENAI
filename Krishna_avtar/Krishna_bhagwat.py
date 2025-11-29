from dotenv import load_dotenv
load_dotenv(override=True)
import tiktoken
import numpy as np
from codecs import utf_8_decode
import os
from dotenv.parser import Reader
from openai import OpenAI
import gradio as gr
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


print(os.getenv("OPENAI_API_KEY"))
from pypdf import PdfReader

groq_api_key = os.getenv('GROQ_API_KEY')

db_name = "bhagwat_db"




pdf_path = "/Users/apple/Abhilash_Projects/Bhagavad-Gita-For-Awakeninging-full-2-2021.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Divided into {len(chunks)} chunks")
print(f"First chunk:\n\n{chunks[0]}")


groq_url = "https://api.groq.com/openai/v1"
groq = OpenAI(api_key=groq_api_key, base_url=groq_url)

name="Krishnavtar"
with open("summary_avtar.txt",'r',encoding="utf-8") as f :
    summary_avtar=f.read()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)


system_prompt = f"You are acting as {name}. You are answering questions based on learning from bhagwat, \
particularly questions related to anyone way of living. \
Your responsibility is to represent {name} for interactions on the chat as clearly as possible. \
You are given a summary in form of summary_avtar.txt of {name}'s  which you can use to answer questions. \
Be professional and engaging, as you are representing almighty. \
If you don't know the answer, say so"

system_prompt += f"\n\n## Summary_avtar:\n{summary_avtar}\n\n## Bhagwat\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."



retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # tune k as you like
def get_context_from_bhagwat(query: str) -> str:
    docs = retriever.invoke(query)
    if not docs:
        return "No specific passages found in the Bhagwat for this question."
    return "\n\n---\n\n".join(d.page_content for d in docs)



def strip_extras(obj):
    """Recursively strip any keys except 'role' and 'content' from dicts."""
    if isinstance(obj, dict):
        return {k: strip_extras(v) for k, v in obj.items() if k in ['role', 'content']}
    elif isinstance(obj, list):
        return [strip_extras(item) for item in obj]
    else:
        return obj  # str, None, etc.

def chat(message, history):
    # Handle if history has tuples (old Gradio format) or mixed
    clean_history = []
    for item in history:
        if isinstance(item, tuple):  # e.g., ("user", "text")
            clean_msg = {"role": "user", "content": item[0]} if len(item) > 0 else {} 
        elif isinstance(item, dict):
            clean_msg = strip_extras(item)  # Strips metadata deeply
        else:
            continue  # Skip junk like None
        if 'role' in clean_msg and 'content' in clean_msg:
            clean_history.append(clean_msg)
    
    # Double-check: Ensure no extras snuck in
    for msg in clean_history:
        msg_keys = set(msg.keys())
        if msg_keys != {'role', 'content'}:
            print(f"WARNING: Extra keys in msg: {msg_keys}")
            msg.clear()
            msg.update({"role": msg.get("role"), "content": msg.get("content")})
    
    context = get_context_from_bhagwat(message)
    system_with_context = (
        system_prompt
        + "\n\n## Relevant excerpts from Bhagwat (retrieved by your divine memory):\n"
        + context
        + "\n\nAnswer the user's question based ONLY on this context and your summary_avtar persona."
    )
    messages = [{"role": "system", "content": system_with_context}] + clean_history + [{"role": "user", "content": message}]
    
    response = groq.chat.completions.create(
        model="openai/gpt-oss-120b",  # ← Use a real Groq model (not "openai/gpt-oss-120b")
        messages=messages
    )
    
    return response.choices[0].message.content

gr.ChatInterface(chat, type="messages").launch(share=True)