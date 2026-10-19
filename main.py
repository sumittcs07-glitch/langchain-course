from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

tavily= TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0)
tools=[search]
agent=create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in tokyo")})
    print(result)
if __name__ == "__main__":
    main()
