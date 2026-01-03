from dotenv import load_dotenv
from duckduckgo_search import DDGS
from serpapi import GoogleSearch

import os
load_dotenv()

# Check if API key is set
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("Error: ANTHROPIC_API_KEY environment variable is not set.")
    print("Please set your API key in the .env file or environment variables.")
    print("\nExample .env file:")
    print("ANTHROPIC_API_KEY=your_api_key_here")
else:
    from langchain_anthropic import ChatAnthropic
    
    # Initialize the LLM
    llm = ChatAnthropic(model="claude-opus-4-5-20251101")


from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from serpapi import GoogleSearch

# ---- live search (RAG layer) ----


api_key = os.getenv("SERPAPI_API_KEY")
print("SERPAPI_API_KEY:", api_key)

print("ANTHROPIC_API_KEY seen by app22:", os.getenv("ANTHROPIC_API_KEY"))



def live_search(query: str, max_results: int = 5) -> str:
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.environ["SERPAPI_API_KEY"],
        "num": max_results,
    }
    
    results = GoogleSearch(params).get_dict()

    snippets = []
    for r in results.get("organic_results", []):
        snippets.append(
            f"{r.get('title')} ({r.get('date','')}): {r.get('snippet')}"
        )

    return "\n".join(snippets)

# ---- LangGraph state ----
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ---- chatbot node ----
def chatbot(state: State) -> State:
    user_message = state["messages"][-1].content
    search_context = live_search(user_message)

    prompt = f"""
You MUST answer ONLY using the search results below.
If the information is missing or uncertain, say "I don't know".

Search results:
{search_context}

Question:
{user_message}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# ---- build graph ----
builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)
builder.add_edge(START, "chatbot_node")
builder.add_edge("chatbot_node", END)

graph = builder.compile()