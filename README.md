# Multi-Agent-RAG
A Retrieval-Augmented Generation (RAG) pipeline with multi-agent LLM orchestration, cross-encoder reranking, and Chroma vector search. Features advanced document retrieval, query filtering, and LLM-based conversational AI for knowledge-intensive applications. 🚀

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline with multi-agent orchestration, cross-encoder reranking, LLM-based conversational retrieval, and content filtering for enhanced accuracy and cost efficiency.

The system integrates:

- Vector Database (Chroma DB) for efficient document storage & retrieval.
- Cross-Encoder Reranking Model for improving search relevance.
- Multi-Agent LLM Selection (GPT-4o-mini for general queries, GPT-01-mini for precise metric-based responses).
- Pre-processing & Post-processing Guardrails for query validation, hallucination detection, and content filtering.
- Token Usage Monitoring & Cost Tracking.
- Docker Containerization & Deployment on GCP.

Features
- Conversational Retrieval Chain using LangChain.
- Multi-Agent LLM Routing based on query intent.
- Document Chunking & Vector Search for efficient retrieval.
- Cross-Encoder-Based Reranking for improved precision.
- Toxicity & Off-Topic Classification to ensure query relevance.
- Comprehensive Observability & Logging for debugging & auditing.
- Containerized Deployment with Docker for easy scalability.

## Tech Stack
- Programming Language: Python
- LLM Framework: LangChain
- Embedding Models: HuggingFace (all-MiniLM-L6-v2)
- Vector Database: Chroma DB
- LLM APIs: OpenAI GPT-4o-mini, GPT-01-mini
- Reranking Model: Transformer-based Cross-Encoder
- UI Framework: Gradio
- Toxicity & Zero-Shot Classification: HuggingFace Pipelines
- Logging & Monitoring: JSON-based logging system
- Containerization: Docker
- Cloud Deployment: Google Cloud Platform (GCP)

## Installation & Setup
1. Clone the Repository
git clone https://github.com/yourusername/GenAI-RAG-Reranker.git

cd GenAI-RAG-Reranker

2. Install Dependencies
Ensure you have Python 3.9+ installed. Then, install the required dependencies:
pip install -r requirements.txt

3. Set Up Environment Variables
Create a .env file in the root directory and add the following:

OPENAI_API_KEY=your-openai-api-key
EMBEDDINGS_MODEL_NAME=D:/Embedding_Models
ZERO_SHOT_MODEL_DIR=D:/Deep_Models
TOXICITY_MODEL_DIR=D:/Unitary_Models
CROSS_ENCODER_DIR=D:/Deep_Models/Rerank

4. Run the Application
python app.py

## Usage Guide
Start a Retrieval Session
Run the script and select an inquiry category:

option1 : Products/Employees
option2: Contracts/Company

Enter your Employee ID, Name, and Email.

Start asking queries. The system:

Filters off-topic & toxic queries.
Retrieves relevant documents.
Reranks the top results using a cross-encoder.
Routes the query to an appropriate LLM agent.
Returns an optimized response with source references.
Type 'quit' to end the session.

Docker Deployment
To run the project inside a Docker container:

Build the Docker image:
docker build -t genai-rag-app .

Run the container:
docker run -p 5000:5000 genai-rag-app
