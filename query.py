import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


# --- Calculator Tool (guarded) ---
@tool
def calculator(expression: str) -> str:
    """Safely evaluate math expressions like '23*57' or '(7+9)/2'. 
    Use this only for numeric calculations."""
    import re
    if not re.match(r"^[0-9+\-*/(). ]+$", expression):
        return "❌ Unsafe or invalid math expression."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def main():
    load_dotenv()

    # 1. Load embeddings + Chroma DB
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever()

    # 2. Wraped the retriever as a Tool
    rag_tool = Tool(
        name="PDF Retriever",
        func=lambda q: retriever.get_relevant_documents(q),
        description="Use this tool to answer questions strictly about the ingested PDF documents."
    )

    # 3. Safe Web Search Tool
    duckduckgo = DuckDuckGoSearchRun()

    @tool
    def safe_web_search(query: str) -> str:
        """Use this tool for questions about current events, recent facts, or external knowledge 
        NOT available in the PDF. Avoid unsafe or illegal queries."""
        blocked = ["hack", "illegal", "nsfw", "bomb", "weapon"]
        if any(b in query.lower() for b in blocked):
            return "❌ This query is blocked for safety reasons."
        return duckduckgo.run(query)

    # 4. Conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 5. Define the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 6. Initialized Agent with multiple tools
    tools = [rag_tool, calculator, safe_web_search]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

    # 7. Chat loop
    print("🤖 Agentic RAG Chatbot with Auto Tool Selection is ready. Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent.run(query)
        print("Bot:", response)


if __name__ == "__main__":
    main()
