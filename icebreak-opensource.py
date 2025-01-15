#ollama, llama3, mistral arre the open source llm
# ---commands---
# ollama run llama3
#--- langchain package---
# pip install langchain-ollama


import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

information = """
Jeffrey Preston Bezos (/ˈbeɪzoʊs/ BAY-zohss;[2] né Jorgensen; born January 12, 1964) is an American businessman best known as the founder, executive chairman, and former president and CEO of Amazon, the world's largest e-commerce and cloud computing company. He is the second wealthiest person in the world, with a net worth of US$251 billion as of December 17, 2024, according to Forbes and Bloomberg Billionaires Index.[3] He was the wealthiest person from 2017 to 2021, according to both the Bloomberg Billionaires Index and Forbes.[4][5]
Bezos was born in Albuquerque and raised in Houston and Miami. He graduated from Princeton University in 1986 with degrees in electrical engineering and computer science. He worked on Wall Street in a variety of related fields from 1986 to early 1994. Bezos founded Amazon in mid-1994 on a road trip from New York City to Seattle. The company began as an online bookstore and has since expanded to a variety of other e-commerce products and services, including video and audio streaming, cloud computing, and artificial intelligence. It is the world's largest online sales company, the largest Internet company by revenue, and the largest provider of virtual assistants and cloud infrastructure services through its Amazon Web Services branch.
Bezos founded the aerospace manufacturer and sub-orbital spaceflight services company Blue Origin in 2000. Blue Origin's New Shepard vehicle reached space in 2015 and afterwards successfully landed back on Earth; he flew into space on Blue Origin NS-16 in 2021. He purchased the major American newspaper The Washington Post in 2013 for $250 million and manages many other investments through his venture capital firm, Bezos Expeditions. In September 2021, Bezos co-founded Altos Labs with Mail.ru founder Yuri Milner.[6]
"""

if __name__ == '__main__':
    print("Hello langchain")
    # print(os.environ['OPENAI_API_KEY'])

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting fact about them
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm = ChatOllama(model_name="llama3") # llm = ChatOllama(model_name="mistral")

    chain = summary_prompt_template | llm

    # So we want to create a chain. And the chain is going to be the summary template followed by a pipe operator into the LM.
    # And this pipe operator comes from the link chain expression language. We're going to review in depth in this course.
    # But you can really simplify it by thinking that it simply making an API call to OpenAI.

    res = chain.invoke(input={"information": information})

    print(res)