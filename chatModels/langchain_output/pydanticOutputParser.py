from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import os 
import logging
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN") 
if not token :
     raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not found in environment") 


class Person(BaseModel):
     
      name : str = Field(description = 'name of the person')
      age : int = Field(description="age of the person", gt=0, lt = 100)
      city : str =Field(description='city of the person')




try:
     
     llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1",
        task="text-generation",
        max_new_tokens=1000,
        temperature=0.0,
    )
     
     model = ChatHuggingFace(llm=llm)

     parser = PydanticOutputParser(pydantic_object=Person)

     template = PromptTemplate(
          template = 'Generate the name, age and city of a fictional {place} person \n {format_instruction}',
          input_variables=['place','format_instruction']
     )

     format_instructions = parser.get_format_instructions()

     chain = template | model | parser

     response = chain.invoke({
        "place": "India",
        "format_instruction": format_instructions
    })
     print(template)
     print("\n")
     print(response)
     print("\n")
     chain.get_graph().print_ascii()
           


except Exception as e:
    logging.exception("LLM invocation failed")
    raise