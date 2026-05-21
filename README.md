📄 Research Paper Assistant (RAG-Based AI Application)

An AI-powered Research Paper Assistant built using Retrieval-Augmented Generation (RAG) that allows users to upload research papers in PDF format and interact with them through natural language queries. The application provides intelligent summaries, semantic search, metadata extraction, keyword generation, and contextual question-answering using Large Language Models (LLMs).

The system is developed using Streamlit, LangChain, ChromaDB, HuggingFace embeddings, and Groq LLMs to create an interactive AI research assistant capable of understanding and analyzing scientific documents efficiently.
https://research-paper-summarizer-26.streamlit.app/
🚀 Features
📑 Upload and analyze research papers in PDF format
🤖 RAG-based conversational AI for paper-specific Q&A
🔍 Semantic search using vector embeddings
🧠 AI-generated summaries and insights
🏷️ Automatic keyword and metadata extraction
💬 Interactive chatbot with conversational memory
📊 Deep insights including methodologies, datasets, equations, and limitations
📥 Export summaries and chat history
🌐 Deployed with Streamlit for real-time usage
🛠️ Tech Stack
Python
Streamlit
LangChain
ChromaDB
HuggingFace Embeddings (all-MiniLM-L6-v2)
Groq LLM API
PyPDFLoader
Conversational Retrieval Chain (RAG Architecture)
⚙️ How It Works
User uploads a research paper PDF
PDF text is extracted and split into chunks
Chunks are converted into vector embeddings
Embeddings are stored in ChromaDB vector database
User queries are matched semantically using RAG retrieval
Groq LLM generates context-aware answers and summaries
🎯 Applications
Research paper analysis
Academic assistance
Literature review automation
AI-powered document understanding
Intelligent PDF chatbot systems
📌 Future Improvements
Multi-PDF support
Citation generation
Research paper comparison
Voice-based interaction
Advanced visualization dashboard
Cloud database integration
👨‍💻 Author

Developed by Ankit as an AI/LLM-based research assistant project using modern RAG architecture and conversational AI techniques.
