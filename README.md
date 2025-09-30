# Agentic RAG with ChromaDB

An intelligent chatbot that combines Retrieval-Augmented Generation (RAG) with LangChain agents and content moderation.

## Features

- 📚 PDF document retrieval using ChromaDB vectorstore
- 🔍 Dual web search capabilities (DuckDuckGo and Tavily)
- 🧮 Built-in calculator functionality
- 🛡️ Content moderation using OpenAI
- 💭 Conversation memory for context retention

## Prerequisites

- Python 3.8+
- OpenAI API key
- Tavily API key (optional)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hrshit414194/Agentic_RAG_chromaDB.git
   cd Agentic_RAG_chromaDB
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix/MacOS
   # or
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Usage

1. Add your PDF document as `Chunking_RAG.pdf`
2. Ingest the document:
   ```bash
   python ingest.py
   ```

3. Start the chatbot:
   ```bash
   python query.py
   ```

### Available Commands

- `switch to tavily`: Switch to Tavily search provider
- `switch to duckduckgo`: Switch to DuckDuckGo search provider
- `current provider`: Check current search provider
- `exit`: Quit the application

## License

MIT License - see LICENSE file for details
