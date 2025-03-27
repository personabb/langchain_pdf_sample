
import os
import base64
from io import BytesIO
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv, find_dotenv

#_ = load_dotenv(find_dotenv())
#api_key = os.getenv("OPENAI_APIKEY")

api_key = os.getenv("XAI_API_KEY", "")
api_endpoint = "https://api.x.ai/v1"
print(api_key)

os.environ['OPENAI_API_KEY'] = api_key
os.environ['OPENAI_API_BASE'] = api_endpoint


if api_key == "":
    raise ValueError("OpenAI API Key が設定されていません。")

"""model = OpenAI(
    model="grok-2-image",
    openai_api_key=api_key,
    openai_api_base=api_endpoint,
    #max_tokens=CONFIG["rag_llm_max_output_tokens"],
    #temperature=temperature,
    #top_p=top_p if top_p else 1.0
)"""

model = DallEAPIWrapper(
    model="grok-2-image",
    quality=None,
    size=None,
    style=None,
)




response = model.run("A cat in a tree")

print(response)

print(response.data[0].url)