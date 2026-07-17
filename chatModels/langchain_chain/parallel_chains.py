from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import Field,BaseModel
from typing import Literal
import logging
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableBranch, RunnablePassthrough, RunnableSequence
import os
load_dotenv()






try:


    gemini_model = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest",   
          temperature=0.0 )
    
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1",
        task="text-generation",
        max_new_tokens=1000,
        temperature=0.0,
    )

    model = ChatHuggingFace(llm = llm)

    strParser = StrOutputParser()

    

    prompt1 = PromptTemplate(
        template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
    )

    prompt2 = PromptTemplate(
     template='Generate 5 short question answers from the following text \n {text}',
     input_variables=['text']
)

    prompt3 = PromptTemplate(
     template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)
    notes =  prompt1 | model | strParser,
    branch_chain = RunnableParallel({
        'notes' : RunnablePassthrough(),
        'quiz' : prompt2 | gemini_model | strParser}
    )
    
    #chain =notes |  branch_chain | prompt3 | gemini_model | strParser
    chain = RunnableSequence(
    notes,
    branch_chain,
    prompt3,
    gemini_model,
    strParser
)

    text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

    result = chain.invoke({'text':text})
    print(result,"\n")
    chain.get_graph().print_ascii()





except Exception as e:
    logging.exception("LLM invocation failed")
    raise 

