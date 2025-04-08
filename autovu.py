import pandas as pd

from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities import GoogleSerperAPIWrapper
from tqdm import tqdm


tqdm.pandas()

load_dotenv()

serp_key = os.getenv('SERPER_API_KEY')
uni_api_key = os.getenv('UNI_API_KEY')

# Initialize search tool
search = GoogleSerperAPIWrapper()





llm = ChatOpenAI(
    openai_api_base="https://llms-inference.innkube.fim.uni-passau.de",
    api_key=uni_api_key,
    model = "deepseekr1",
    temperature=0.6
)


def generate_answer(query):
    
    custom_prompt = PromptTemplate.from_template(
        "You are an AI assistant. Your task is to answer the question directly using the latest information which will be provided."
        "\nNote that search results are more up-to-date and reliable than the information you have as we are in 2025."
        "\nAnswer briefly and directly, return only the answer label."
        "\nReturn a confidence score based on the provided context: high, medium or low."
        "\n Your answer should be in the format: Answer: your_answer, Confidence: your_confidence_score."
        "\nUser Question: {input}"
        "\nSearch results: {search}"
    )

    # Create an LLMChain instead of an agent
    llm_chain = LLMChain(
        llm=llm,
        prompt=custom_prompt,
        verbose=True
    )
    # Get the latest factual information first
    search_result = search.results(query)
        
    formatted_results = "\n".join(
        f"Title: {res.get('title', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}\nDate: {res.get('date', 'N/A')}\n"
        for res in search_result['organic']
    )
    # Pass the query to the agent
    try:
        result = llm_chain.run({"input": query, "search": formatted_results})
        print(f"\nFinal Result: {result.strip()}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return result.strip()



if __name__=='__main__':
    
    df = pd.read_csv('company_ceo_31-03.csv')
    df['prediction']=df['companyLabel'].progress_apply(lambda x: generate_answer(f"Who is the current CEO of {x}?"))
    df.to_csv('company_ceo_results.csv')