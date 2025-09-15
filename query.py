import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    PERSIST_DIR = "persist_chroma"
    COLLECTION = "pdf_docs"

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that answers questions using the provided context. "
         "Always be concise and cite relevant sections."),
        ("human", "Question: {input}\n\nContext:\n{context}")
    ])

    # Use correct chain constructors
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    chat_history = []

    print("✅ Ready! Ask me about your PDF. (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        res = retrieval_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })

        answer = res["answer"]
        print("Bot:", answer, "\n")

        chat_history.append(("user", query))
        chat_history.append(("assistant", answer))


if __name__ == "__main__":
    main()
