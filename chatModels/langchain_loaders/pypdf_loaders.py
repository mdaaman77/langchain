from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


try:

    model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest",temperature=0.0)

    loader = PyPDFLoader("./langchain/chatModels/INTERNSHIP_FINAL.pdf")
    docs = loader.load()

    prompt = PromptTemplate(
        template='Generate short and simple notes in 10 lines from the following text \n {text}',
        input_variables=['text']
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    response = chain.invoke({"text": docs})
    print(response)

except Exception as e:
    print("Error initializing models:", e)