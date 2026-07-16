from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, TypedDict, Optional,Literal, Annotated
import os
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not found in environment")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

class Review(TypedDict):
        summary : Annotated[str,"The summary of the review"]
        key_points: Annotated[List[str],"The key points of the content and specifications"]
        pros: Annotated[Optional[List[str]],"The pros of the content and specifications , if not avaliable dont do"]
        cons: Annotated[Optional[List[str]],"The cons of the content and specifications , if not avaliable dont do"]
        sentiments: Annotated[Literal["pos","neg"],"The sentiment of the content and specifications"]

chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that summarizes product content and specifications, and provides key points, pros, cons, and sentiment analysis."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    (
        "human",
        "{query}"
    ),
])
store_messages = []


try:

   

    llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.0,
 
)

    chat_model = ChatHuggingFace(llm=llm)

    messages = "what is the capital of India?"
    prompt = chat_prompt.invoke({'chat_history': store_messages, 'query': messages})
    store_messages.append(messages)

    strucutre = chat_model.with_structured_output(Review)

    response = strucutre.invoke(prompt)
    store_messages.append(response)

    print(response)
    print(store_messages)

except Exception as e:
    logging.exception("LLM invocation failed")
    raise