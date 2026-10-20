from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")

    # FIX: Wrapped HumanMessage in brackets [ ]
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
                )
            ]
        }
    )

    # Because you used structured output, you can print specific fields!
    print("Answer:", result["structured_response"].answer)
    print("Sources:", result["structured_response"].sources)


# FIX: Added the execution block so Python actually runs the code
if __name__ == "__main__":
    main()