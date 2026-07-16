from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate
from typing import List, TypedDict, Optional,Literal, Annotated
from langchain_core.output_parsers import StrOutputParser


import os

from dotenv   import load_dotenv
load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN") 

if not token:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not found in environment") 


try:
 

    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1",
        task="text-generation",
        max_new_tokens=1000,
        temperature=0.0,
    )

    model = ChatHuggingFace(llm=llm)
    #chat_model = model.with_structured_output(Review);

    template1 = PromptTemplate(
        input_variables=["query"],
        template="give me detail expalination on  {query}"
    )

    template2 = PromptTemplate(
        input_variables=["query"],
        template="give me a short summmary on  {query}"
    )
    parser = StrOutputParser()
    chain = template1 | model  | parser| template2 | model | parser

    response =  chain.invoke({"query": "explain mango fruit in detail"})
    print("Chain invoked successfully", response)

except Exception as e:
    logging.exception("LLM invocation failed")
    raise

