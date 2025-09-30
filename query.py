import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch


# Config file for persistence
CONFIG_FILE = "provider_config.txt"


def load_provider():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return f.read().strip()
    return "duckduckgo"


def save_provider(provider: str):
    with open(CONFIG_FILE, "w") as f:
        f.write(provider)


# Initialize providers
duckduckgo = DuckDuckGoSearchRun()
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if TAVILY_API_KEY:
    tavily = TavilySearch(max_results=5, api_key=TAVILY_API_KEY)
else:
    tavily = None

SEARCH_PROVIDER = load_provider()

# Initialize OpenAI client for moderation
client = OpenAI(api_key=OPENAI_API_KEY)


def is_safe_prompt(prompt: str) -> (bool, str):
    """
    Uses OpenAI Moderation API to check if a prompt is safe.
    Returns (is_safe, warning_message).
    """
    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=prompt
        )
        result = response.results[0]
        if result.flagged:
            return False, "⚠️ Your query may contain unsafe or illegal content. Please rephrase and try again."
        return True, ""
    except Exception as e:
        # Fallback: treat as safe if moderation API fails
        return True, f"⚠️ Moderation check failed: {e}"


# Calculator Tool
@tool
def calculator(expression: str) -> str:
    """Safely evaluate math expressions like '23*57' or '(7+9)/2'."""
    import re
    if not re.match(r"^[0-9+\-*/(). ]+$", expression):
        return "❌ Unsafe or invalid math expression."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# Web Search Tool (still wrapped with guardrails for safety) 
@tool
def safe_web_search(query: str) -> str:
    """Safely search the web for real-time information."""

    global SEARCH_PROVIDER
    if len(query) > 200:
        return "❌ Query too long, please shorten it."

    try:
        if SEARCH_PROVIDER == "duckduckgo":
            result = duckduckgo.run(query)
        elif SEARCH_PROVIDER == "tavily":
            if not tavily:
                return "❌ Tavily selected but no API key found in .env"
            result = tavily.run(query)
        else:
            return "❌ Invalid search provider."
    except Exception as e:
        return f"⚠️ Error while searching: {e}"

    if not result or len(str(result).strip()) == 0:
        return "⚠️ No safe results found."

    if isinstance(result, list):
        safe_results = [r.get("content", "") for r in result if "content" in r]
        return " | ".join(safe_results[:3])
    return str(result).strip()[:500]


def main():
    global SEARCH_PROVIDER
    print(f"🔎 Starting with search provider: {SEARCH_PROVIDER}")

    # Load embeddings + Chroma DB
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # Warn if DB empty
    if vectordb._collection.count() == 0:
        print("⚠️ No documents found in ChromaDB.")
        print("👉 Run `python ingest.py` first to ingest Chunking_RAG.pdf.\n")

    retriever = vectordb.as_retriever()

    # PDF Retriever tool
    rag_tool = Tool(
        name="PDF Retriever",
        func=lambda q: retriever.get_relevant_documents(q),
        description="Use this tool to answer questions strictly about the ingested PDF (Chunking_RAG.pdf)."
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [rag_tool, calculator, safe_web_search]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

    print("🤖 Agentic RAG Chatbot with Moderation is ready. Type 'exit' to quit.")
    print("💡 Commands: 'switch to tavily', 'switch to duckduckgo', 'current provider'\n")

    while True:
        try:
            query = input("You: ")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break

        if query.lower() in ["exit", "quit"]:
            break

        # Moderation check BEFORE agent
        safe, warning = is_safe_prompt(query)
        if not safe:
            print(f"Bot: {warning}")
            continue

        # Commands
        if query.lower() == "switch to tavily":
            SEARCH_PROVIDER = "tavily"
            save_provider(SEARCH_PROVIDER)
            print("🔄 Switched to Tavily search provider.\n")
            continue
        elif query.lower() == "switch to duckduckgo":
            SEARCH_PROVIDER = "duckduckgo"
            save_provider(SEARCH_PROVIDER)
            print("🔄 Switched to DuckDuckGo search provider.\n")
            continue
        elif query.lower() == "current provider":
            print(f"🔎 Current search provider: {SEARCH_PROVIDER}\n")
            continue

        # Run agent
        response = agent.run(query)
        print("Bot:", response)


if __name__ == "__main__":
    main()
