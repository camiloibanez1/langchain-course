from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch

# tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that seaches over internet
    Arts:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    # return "Tokyo weather is sunny and warm."
    return tavily.search(query=query)


llm = ChatOllama(model="gpt-oss:20b")
# llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in Tokyo?")}
    )
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
