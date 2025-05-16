import pandas as pd

from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities import GoogleSerperAPIWrapper
from tqdm import tqdm
from openai import OpenAI
import time


tqdm.pandas()

load_dotenv()




serp_key = os.getenv('SERPER_API_KEY')
uni_api_key = os.getenv('UNI_API_KEY')
openai_key = os.getenv('OPENAI_API_KEY')

# Initialize search tool
search = GoogleSerperAPIWrapper()
client = OpenAI(api_key=openai_key)





llm = ChatOpenAI(
    openai_api_base="https://llms-inference.innkube.fim.uni-passau.de",
    api_key=uni_api_key,
    model = "deepseekr1",
    temperature=0.6
)

def answer_no_web(query):
    messages = [
        SystemMessage(
            content="You are an AI assistant. Your task is to answer the question directly based on your knowledge."
            "\nAnswer briefly and directly, return only the answer label."
            "\nReturn a confidence score based on your certainty: high, medium or low."
            "\nYour answer should be in the format: Answer: your_answer, Confidence: your_confidence_score."
        ),
        HumanMessage(
            content=f"Answer the following question: {query}"
        ),
        ]
    response = llm(messages)
    return response.content.strip()
    
    

def answer_with_gpt(query):
    response = client.responses.create(
        model="gpt-4o-mini",
        tools=[{
            "type": "web_search_preview",
            "search_context_size": "low",
        }],
        # input=f"Answer the following question: {query} Search the web if you are not sure about the answer."
        # ,
        input=[
        {
         "role": "developer",
        "content": (
        "You are an information extraction assistant. You must always respond in this exact format:\n"
        "Answer: <answer here, return only the name>\n"
        "Confidence: <high | medium | low>\n"
        "Reference: <a relevant reference link>\n\n"
        "Do not include any other text, commentary, or formatting."
        )
        },
        {
            "role": "user",
            "content": query
        }
        ])
    return response.output_text
    
    
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
        #verbose=True
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
        #print(f"\nFinal Result: {result.strip()}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
    return result.strip()

def incremental_save():

    df = pd.read_csv('data/company_ceo_31-03.csv')
    df['prediction'] = None 

    save_path = 'company_ceo_31-03_progress.csv'
     
    try:
        df_existing = pd.read_csv(save_path)
        df.update(df_existing)
        print("Resumed from existing saved file.")
    except FileNotFoundError:
        pass

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if pd.isna(df.at[idx, 'prediction']):
            try:
                prompt = f"Who is the current CEO of {row['companyLabel']}?"
                prediction = answer_no_web(prompt)
                df.at[idx, 'prediction'] = prediction

                # Save after every N rows to avoid data loss
                if idx % 5 == 0:  # every 5th row
                    df.to_csv(save_path, index=False)
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                time.sleep(2)  # slight delay if needed

   
    df.to_csv(save_path, index=False)


if __name__=='__main__':
    
    #incremental_save()
    
    df = pd.read_csv('data/international_organizations_leadership_31-03.csv')
    #df['prediction']=df['governorLabel'].progress_apply(lambda x: answer_with_gpt(f"Who is the current governor of {x}?"))
    # #df['prediction']=df['teamLabel'].progress_apply(lambda x: generate_answer(f"What is the current global ranking of {x}?"))
    df['prediction']=df['organizationLabel'].progress_apply(lambda x: answer_no_web(f"Who is the current leader of {x}?"))
    df.to_csv('international_organizations_leadership_31-03_results_r1_no_web.csv', index=False)