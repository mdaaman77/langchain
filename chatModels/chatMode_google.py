import logging
import os
from dotenv import load_dotenv
from typing import List, TypedDict, Optional, Literal, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logging.basicConfig(level=logging.INFO)

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY not found")


class Review(TypedDict):
    summary: Annotated[str, "Summary"]
    key_points: Annotated[List[str], "Key points"]
    pros: Annotated[Optional[List[str]], "Pros"]
    cons: Annotated[Optional[List[str]], "Cons"]
    sentiments: Annotated[Literal["pos", "neg"], "Sentiment"]


chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You summarize content into structured review format"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

store_messages = []

try:
    model = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",   
        temperature=0.0
    )

    structured_model = model.with_structured_output(Review)

    query = "India is a large country with capital New Delhi and growing economy."

    prompt = chat_prompt.invoke({
        "chat_history": [],
        "query": query
    })

    response = structured_model.invoke(prompt)

    print(response)

except Exception:
    logging.exception("LLM invocation failed")
    raise