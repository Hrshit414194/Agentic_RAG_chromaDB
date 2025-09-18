import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.tools import tool


# Define a calculator tool
@tool
def calculator(expression: str) -> str:
    """Accurately evaluates math expressions. 
    Example: '23 * 57' or '(7 + 9) / 2'"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def main():
    load_dotenv()

    # Loaded embeddings + Chroma DB
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever()

    # Wrap retriever as a Tool
    rag_tool = Tool(
        name="PDF Retriever",
        func=lambda q: retriever.get_relevant_documents(q),
        description="Use this tool to fetch relevant information from the uploaded PDF."
    )

    # Loaded conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Define the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Initialized the Agent with multiple tools
    tools = [rag_tool, calculator]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

    # Chat loop
    print("🤖 Agentic RAG Chatbot is ready, Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent.run(query)
        print("Bot:", response)


if __name__ == "__main__":
    main()
