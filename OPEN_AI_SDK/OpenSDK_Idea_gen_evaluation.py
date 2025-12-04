from http import client
from agents import Agent, agent,Runner,OpenAIChatCompletionsModel
from dotenv import load_dotenv
import os
from openai import Client, OpenAI
from IPython.display import Markdown, display
load_dotenv(override=True)
from openai import AsyncOpenAI  # Async client import



openai_api_key = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
 

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if groq_api_key:
    print(f"Groq API Key exists and begins {groq_api_key[:4]}")
else:
    print("Groq API Key not set (and this is optional)")


groq_url = "https://api.groq.com/openai/v1"
 

#deepseek = OpenAI(api_key=deepseek_api_key, base_url=deepseek_url)
groq = OpenAI(api_key=groq_api_key, base_url=groq_url)
#grok = OpenAI(api_key=grok_api_key, base_url=grok_url)
openai=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


System_prompt = f"""You are a helpful assistant that generates best business idea from technology sector to be presented in a competion.

The response should be in the following format:
Business Idea: <business_idea>
Business Model: <business_model>
Business Plan: <business_plan>
Business Strategy: <business_strategy>
Business Implementation: <business_implementation>
Business Evaluation: <business_evaluation>
Business Conclusion: <business_conclusion>
The response should not include any markup"""

User_prompt = f"""Generate a best business idea from technology sector to be presented in a competion."""


groq = AsyncOpenAI(api_key=groq_api_key, base_url=groq_url)



        

oss_model="openai/gpt-oss-120b"
gpt_model="gpt-5-nano"


groq_model = OpenAIChatCompletionsModel(
    model=oss_model,  # Or mixtral-8x7b-32768, gemma2-9b-it, etc.
    openai_client=groq
)


gpt_idea_gen_agent=Agent("gpt model",model=gpt_model,instructions=System_prompt)
oss_idea_gen_agent=Agent("oss model",model=groq_model,instructions=System_prompt)



gpt_business_idea=Runner.run_sync(gpt_idea_gen_agent,User_prompt)
oss_business_idea=Runner.run_sync(oss_idea_gen_agent,User_prompt)

print(gpt_business_idea.final_output)
print("-----------------")
print(oss_business_idea.final_output)

judge_prompt=f"""you are the judge whose job is to impartially judge the the best idea generated based on inputs from different models
You recieve the inputs from model_gpt as : {gpt_business_idea} and model_oss as : {oss_business_idea} . Please provide the fair judgement of the ideas in below format

Winner idea : <Winner idea>
Reason for this judgement= <Judgement Reason> """

Judge_agent=Agent("Judge",model=groq_model,instructions=judge_prompt)

final_winner=Runner.run_sync(Judge_agent,"What is the final judgement")
print(final_winner.final_output)