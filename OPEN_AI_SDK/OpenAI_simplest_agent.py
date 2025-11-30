from agents import Runner,Agent
import os

from dotenv import load_dotenv
load_dotenv(override=True)


agents=Agent(name="GK agent",
                 instructions="you are an agent with good General Knowledge" ,
                 model="gpt-4o-mini")


result=Runner.run_sync(agents,"what is the capital of Japan")

print(result.final_output)
