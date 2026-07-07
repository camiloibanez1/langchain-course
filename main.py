from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from tavily import TavilyClient

# tavily = TavilyClient()


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


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


# llm = ChatOllama(model="gpt-oss:20b")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in Tokyo?")}
    )
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
