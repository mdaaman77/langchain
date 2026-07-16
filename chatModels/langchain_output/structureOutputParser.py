from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not found")


try:
    # ---------------- MODEL ----------------
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.0,
    )

    model = ChatHuggingFace(llm=llm)

    # ---------------- SCHEMA ----------------
    schema = [
        ResponseSchema(name="fact_1", description="fact 1 about the topic"),
        ResponseSchema(name="fact_2", description="fact 2 about the topic"),
        ResponseSchema(name="fact_3", description="fact 3 about the topic"),
    ]

    parser = StructuredOutputParser.from_response_schemas(schema)

    format_instructions = parser.get_format_instructions()

    # ---------------- PROMPT ----------------
    template = PromptTemplate(
        template="Give 3 facts about {query}\n{format_instructions}",
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions},
    )

    # ---------------- CHAIN ----------------
    chain = template | model | parser

    response = chain.invoke({"query": "mango fruit"})

    print("Response:\n", response)

except Exception:
    logging.exception("LLM invocation failed")
    raise