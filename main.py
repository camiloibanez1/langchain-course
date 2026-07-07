# import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()  # Load environment variables from .env file


def main():
    print("Hello from langchain-course!")
    # print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
    information = """
    Martin Reisman (February 1, 1930 – December 7, 2012) was an American table tennis player and author.[1] He won the U.S. Men's Singles Championship in 1958 and 1960 and the U.S. Hardbat Championship in 1997.[2] He advocated the traditional hardbat style of table tennis.

    Reisman was active in New York City's table tennis community for decades. He was nicknamed "the Needle" for his quick wit and slender build.[3] In his 1974 memoir The Money Player, he wrote that top table tennis players had to be "gamblers or smugglers."[4]
    """

    summary_template = """
    given the information {information}, about a person I want you to create:
    1. a short summary
    2. two interesting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # llm = ChatOpenAI(
    #     model_name="gpt-5",
    #     temperature=0,
    # )

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

    # llm = ChatOllama(
    #     model="gemma3:270m",
    #     temperature=0
    # )

    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    # print(response.content)
    print(response.content[0]["text"])


if __name__ == "__main__":
    main()
