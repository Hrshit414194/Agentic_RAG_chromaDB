import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults


#Config file for persistence 
CONFIG_FILE = "provider_config.txt"


def load_provider():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return f.read().strip()
    return "duckduckgo"


def save_provider(provider: str):
    with open(CONFIG_FILE, "w") as f:
        f.write(provider)


#Initialized providers 
duckduckgo = DuckDuckGoSearchRun()
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if TAVILY_API_KEY:
    tavily = TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY)
else:
    tavily = None

SEARCH_PROVIDER = load_provider()


#Calculator Tool 
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


#Safe Web Search Tool 
@tool
def safe_web_search(query: str) -> str:
    """Safely search the web for real-time information."""

    global SEARCH_PROVIDER
    blocked = ["hack", "illegal", "nsfw", "bomb", "weapon"]
    if any(b in query.lower() for b in blocked):
        return "❌ This query is blocked for safety reasons."

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

    # Check if database has documents
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

    print("🤖 Agentic RAG Chatbot is ready. Type 'exit' to quit.")
    print("💡 Commands: 'switch to tavily', 'switch to duckduckgo', 'current provider'\n")

    while True:
        try:
            query = input("You: ")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break

        if query.lower() in ["exit", "quit"]:
            break

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

        response = agent.run(query)
        print("Bot:", response)


if __name__ == "__main__":
    main()
