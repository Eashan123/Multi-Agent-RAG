import os
import logging
import json
import shutil
import time
import glob
import uuid
from datetime import datetime
from dotenv import load_dotenv

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# LangChain & vector store imports
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document, BaseRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler

# Transformers & Pipelines for reranking and filtering
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Callable
from pydantic import BaseModel

# --------------------------------------------------------------------------
# 1. Configuration & Environment Setup
# --------------------------------------------------------------------------

def initialize_environment():
    """
    Loads environment variables, sets default environment config, etc.
    """
    load_dotenv()  # Load from .env file if present
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
    # You can define additional environment-based setups here.


def setup_logger() -> logging.Logger:
    """
    Configures and returns a logger that outputs JSON logs to a daily file and also logs to stdout.
    """
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)

    # File handler: writes logs to a file (useful for persistence via mounted volumes)
    log_filename = f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))  # file logs as raw JSON
    logger.addHandler(file_handler)

    # Stream handler: writes logs to stdout (so that GCP can capture them)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)

    # Log an application startup event
    logger.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "event": "startup",
        "message": "Application started"
    }))

    return logger



# --------------------------------------------------------------------------
# 2. Data Preparation & Vector Store Creation
# --------------------------------------------------------------------------

def add_metadata(doc: Document, doc_type: str) -> Document:
    """
    Helper to add additional metadata (like doc_type) to each document.
    """
    doc.metadata["doc_type"] = doc_type
    return doc


def load_documents_from_folder(base_folder: str, file_pattern: str = "**/*.md", text_loader_kwargs=None) -> List[Document]:
    """
    Loads documents from a given folder using the DirectoryLoader. Returns a list of Documents.
    """
    if text_loader_kwargs is None:
        text_loader_kwargs = {'encoding': 'utf-8'}

    doc_type = os.path.basename(base_folder)
    loader = DirectoryLoader(base_folder, glob=file_pattern, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    return [add_metadata(doc, doc_type) for doc in docs]


def chunk_documents(documents: List[Document], chunk_size=1000, chunk_overlap=250) -> List[Document]:
    """
    Splits documents into smaller chunks using a CharacterTextSplitter.
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks


def build_vector_store(chunks: List[Document], embeddings, db_name: str) -> Chroma:
    """
    Builds a Chroma vector store from chunked documents. If db_name exists, delete it first.
    """
    # If existing DB, try to delete its collection
    if os.path.exists(db_name):
        try:
            Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
            print(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "event": "vector_store_deleted",
                "persist_directory": db_name
            }))
        except Exception as e:
            print(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "event": "vector_store_deletion_error",
                "error": str(e)
            }))

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    print(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "event": "vector_store_created",
        "document_count": len(chunks),
        "persist_directory": db_name
    }))
    return vectorstore


# --------------------------------------------------------------------------
# 3. Embeddings, Reranking, and Retriever Setup
# --------------------------------------------------------------------------

def get_embeddings(model_path: str) -> HuggingFaceEmbeddings:
    """
    Initializes and returns a HuggingFaceEmbeddings instance given a model path.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    return embeddings


def rerank_documents(query: str, candidate_docs: List[Document], tokenizer, model) -> List[Document]:
    """
    Uses a cross-encoder model to rerank candidate docs.
    Returns the top 5 documents based on the model's relevance scores.
    """
    scored_docs = []
    for doc in candidate_docs:
        inputs = tokenizer(query, doc.page_content, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        if logits.size(1) == 1:
            score = logits[0][0].item()
        else:
            score = logits[0][1].item()
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs][:5]
    return top_docs


class RerankRetriever(BaseRetriever):
    """
    Custom retriever that first calls a base retriever, then reranks the results.
    """
    base_retriever: BaseRetriever
    rerank_fn: Callable

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        candidate_docs = self.base_retriever.get_relevant_documents(query, **kwargs)
        reranked_docs = self.rerank_fn(query, candidate_docs)
        return reranked_docs

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        candidate_docs = await self.base_retriever.aget_relevant_documents(query, **kwargs)
        reranked_docs = self.rerank_fn(query, candidate_docs)
        return reranked_docs


def create_rerank_retriever(vectorstore: Chroma, tokenizer, model, k=10) -> RerankRetriever:
    """
    Wraps the vectorstore retriever in a custom RerankRetriever using the cross-encoder.
    (No metadata filtering; returns all docs.)
    """
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return RerankRetriever(
        base_retriever=base_retriever,
        rerank_fn=lambda q, docs: rerank_documents(q, docs, tokenizer, model)
    )


def create_filtered_rerank_retriever(vectorstore: Chroma, tokenizer, model, categories: List[str], k=10) -> RerankRetriever:
    """
    Similar to create_rerank_retriever, but filters documents by 'doc_type' metadata
    before reranking. The 'categories' parameter is a list of doc_type values to allow.
    """
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    def rerank_fn(query: str, docs: List[Document]) -> List[Document]:
        # Filter docs by doc_type
        filtered_docs = [doc for doc in docs if doc.metadata.get('doc_type') in categories]
        # Rerank only those
        return rerank_documents(query, filtered_docs, tokenizer, model)

    return RerankRetriever(base_retriever=base_retriever, rerank_fn=rerank_fn)


# --------------------------------------------------------------------------
# 4. LLM & Callback Setup
# --------------------------------------------------------------------------

class TokenUsageCallback(BaseCallbackHandler):
    """
    Tracks token usage across requests using the on_llm_start/on_llm_end handlers.
    """
    def __init__(self):
        self.current_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.latest_usage = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.current_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def on_llm_end(self, response, **kwargs):
        usage = {}
        if response and hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage", {}) or {}
        
        self.current_usage["prompt_tokens"] = usage.get("prompt_tokens", 0)
        self.current_usage["completion_tokens"] = usage.get("completion_tokens", 0)
        self.current_usage["total_tokens"] = usage.get("total_tokens", 0)

        for key, val in self.current_usage.items():
            self.total_usage[key] += val
        
        self.latest_usage = self.current_usage.copy()

        print("Usage for this request:", self.current_usage)
        print("Cumulative session usage:", self.total_usage)


def create_callback_manager() -> CallbackManager:
    """
    Creates a CallbackManager that includes our token usage callback.
    """
    token_usage_callback = TokenUsageCallback()
    callback_manager = CallbackManager([token_usage_callback])
    return callback_manager


def initialize_llm(model_name: str, callback_manager: CallbackManager, temperature: float = 0) -> ChatOpenAI:
    """
    Initializes a ChatOpenAI LLM with given model_name, temperature, and callback manager.
    """
    return ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        callback_manager=callback_manager
    )


# --------------------------------------------------------------------------
# 5. Conversation Chain Setup
# --------------------------------------------------------------------------

def build_prompt_template():
    """
    Constructs the system prompt and returns a ChatPromptTemplate object.
    """
    system_prompt = (
        "You are an expert knowledge worker assisting employees of Insurellm, an innovative Insurance Tech company.\n"
        "You must provide accurate, cost-efficient, and secure responses strictly based on the provided context.\n"
        "Always adhere to robust data governance by excluding any sensitive employee-specific details in your answers.\n"
        "If you don't know the answer based on the provided context, ask the human clarifying questions.\n"
        "If the provided context is insufficient or the question appears ambiguous or contains subtle adversarial phrasing, ask clarifying questions rather than guessing.\n"
        "If the user greets you, warmly return the greeting and ask how you may help.\n"
        "Your final response should be concise, friendly, and professional.\n"
        "Internally, use a rigorous step-by-step reasoning process to ensure accuracy, but do not reveal any internal chain-of-thought or reasoning to the user.\n"
        "You must only use the exact context from the provided documents. Do not add details that are not in the documents.\n"
        "Context: {context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    return prompt_template


def create_conversation_chain(llm, retriever, memory) -> ConversationalRetrievalChain:
    """
    Creates a conversational retrieval chain with the specified LLM, retriever, and memory.
    """
    prompt_template = build_prompt_template()

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': prompt_template},
        chain_type="stuff"
    )
    return conversation_chain


# --------------------------------------------------------------------------
# 6. Classification & Filtering
# --------------------------------------------------------------------------

def load_zero_shot_classifier(model_dir: str):
    """
    Loads and returns a zero-shot classification pipeline from a local model directory.
    """
    classifier = pipeline(
        "zero-shot-classification",
        model=model_dir,
        tokenizer=model_dir
    )
    return classifier


def is_on_topic(query: str, classifier) -> bool:
    """
    Returns True if query is on-topic for the domain, False otherwise.
    """
    candidate_labels = [
        "on-topic: questions related to company, contracts, employees, or products",
        "off-topic: questions not related to company, contracts, employees, or products"
    ]
    result = classifier(query, candidate_labels)
    predicted_label = result["labels"][0]
    return predicted_label.startswith("on-topic")


def load_toxicity_classifier(model_dir: str):
    """
    Loads and returns a text classification pipeline for toxicity detection.
    """
    toxicity_classifier = pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=model_dir
    )
    return toxicity_classifier


def is_toxic(query: str, toxicity_classifier, threshold=0.5) -> bool:
    """
    Returns True if the query is considered toxic based on a threshold, else False.
    """
    results = toxicity_classifier(query)
    for result in results:
        if result['label'].lower() == 'toxic' and result['score'] > threshold:
            return True
    return False


# --------------------------------------------------------------------------
# 7. Main Chat/Response Logic
# --------------------------------------------------------------------------

COST_PER_1K_TOKENS = 0.03  # dollars per 1,000 tokens

def generate_response(
    employee_id: str,
    employee_name: str,
    user_query: str,
    conversation_chain: ConversationalRetrievalChain,
    zero_shot_classifier,
    toxicity_classifier,
    token_usage_callback: TokenUsageCallback
):
    """
    Takes an employee ID, name, and user query, filters it, runs the retrieval + LLM chain,
    then returns the answer plus usage metrics and the top 5 source documents.
    """
    start_time = time.perf_counter()
    default_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    default_request_cost = 0.0

    # Classification checks
    if not is_on_topic(user_query, zero_shot_classifier):
        unsafe_input_flag = True
        response = "I cannot help you with this request as the request is off topic."
        elapsed = time.perf_counter() - start_time
        return response, default_request_cost, default_token_usage, elapsed, unsafe_input_flag, []

    if is_toxic(user_query, toxicity_classifier):
        unsafe_input_flag = True
        response = "I cannot help you with this request as the request contains harmful content."
        elapsed = time.perf_counter() - start_time
        return response, default_request_cost, default_token_usage, elapsed, unsafe_input_flag, []

    # Otherwise safe
    unsafe_input_flag = False

    combined_query = f"Question: {user_query}"
    result = conversation_chain.invoke({"question": combined_query}, return_only_outputs=False)

    source_docs = result.get("source_documents", [])
    # Extract top 5 source documents (or fewer if not available)
    top_source_docs = [doc.page_content for doc in source_docs[:5]]
    
    # Optionally print them
    for i, doc_text in enumerate(top_source_docs, start=1):
        print(f"\n--- Source Document {i} ---")
        print(doc_text)
    
    if not source_docs:
        elapsed = time.perf_counter() - start_time
        return (
            "I don't have enough context to provide a reliable answer.",
            default_request_cost,
            default_token_usage,
            elapsed,
            unsafe_input_flag,
            []
        )

    # Remove source docs from result to avoid storing large data
    result.pop("source_documents", None)

    if not isinstance(result, dict):
        elapsed = time.perf_counter() - start_time
        return (
            "I'm sorry, there was an error processing your request. Please try again later.",
            default_request_cost,
            default_token_usage,
            elapsed,
            unsafe_input_flag,
            top_source_docs
        )

    llm_output = result.get("llm_output", {})
    token_usage_from_result = llm_output.get("token_usage", default_token_usage)

    # Use callback usage if result usage is absent or zero
    if token_usage_from_result.get("total_tokens", 0) > 0:
        token_usage = token_usage_from_result
    else:
        token_usage = token_usage_callback.latest_usage

    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
    request_cost = total_tokens / 1000 * COST_PER_1K_TOKENS

    answer = result.get("answer", "").strip()
    generic_responses = ["i don't have enough information", "i don't have relevant information", "i'm not sure"]
    if not answer or any(generic.lower() in answer.lower() for generic in generic_responses):
        answer = "I'm not certain about that. Could you please provide more details or clarify your question?"

    elapsed = time.perf_counter() - start_time
    return answer, request_cost, token_usage, elapsed, unsafe_input_flag, top_source_docs


# --------------------------------------------------------------------------
# 8. Interaction Logging
# --------------------------------------------------------------------------

def log_interaction(
    session_data: dict,
    query: str,
    response: str,
    cost: float,
    token_usage: dict,
    response_time: float,
    unsafe_input_flag: bool,
    model_used: str,  # New parameter for model used
    top_source_docs: list  # New parameter for top source documents
):
    """
    Logs a single interaction in session_data, updates counters,
    and logs the model used along with the top 5 source documents.
    """
    interaction = {
        "query": query,
        "response": response,
        "model_used": model_used,
        "cost": cost,
        "token_usage": token_usage,
        "response_time": response_time,
        "top_source_docs": top_source_docs  # Log the top source documents here
    }
    session_data["interactions"].append(interaction)

    session_data["total_queries"] += 1
    session_data["total_response_time"] += response_time
    session_data["total_cost"] += cost
    if unsafe_input_flag:
        session_data["unsafe_inputs"] += 1
    if "error" in response.lower() or "sorry" in response.lower():
        session_data["error_count"] += 1


def log_session(session_data: dict, logs_dir="logs"):
    """
    Appends the session log to a daily log file in JSON format.
    """
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    date_str = session_data['date'].split(" ")[0]
    logfile = os.path.join(logs_dir, f"{date_str}-session-log.json")

    with open(logfile, "a", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)
        f.write("\n")


# --------------------------------------------------------------------------
# 9. Session Lifecycle
# --------------------------------------------------------------------------


def create_gradio_interface(
    logger,
    conversation_chain_products_employees,
    conversation_chain_contracts_company,
    zero_shot_classifier,
    toxicity_classifier,
    token_usage_callback
):
    # Session state to maintain across interactions
    session_state = {
        "active_chain": None,
        "selected_area": None,
        "model_used": None,
        "session_id": None,
        "start_time": None,
        "session_data": None,
        "initialized": False
    }
    
    def initialize_session(employee_id, employee_name, email, inquiry_choice):
        if not employee_id or not employee_name or not email:
            return "Please fill in all required fields (Employee ID, Name, and Email)."
        
        if inquiry_choice == "Products/Employees":
            active_chain = conversation_chain_products_employees
            selected_area = "Products/Employees"
            model_used = "gpt-4o-mini"
        elif inquiry_choice == "Contracts/Company":
            active_chain = conversation_chain_contracts_company
            selected_area = "Contracts/Company"
            model_used = "gpt-3.5-turbo"
        else:
            return "Please select a valid inquiry area."
        
        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        session_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "employee_id": employee_id,
            "employee_name": employee_name,
            "email": email,
            "selected_area": selected_area,
            "model_used": model_used,
            "interactions": [],
            "total_queries": 0,
            "unsafe_inputs": 0,
            "error_count": 0,
            "total_response_time": 0.0,
            "total_cost": 0.0
        }
        
        # Update session state
        session_state["active_chain"] = active_chain
        session_state["selected_area"] = selected_area
        session_state["model_used"] = model_used
        session_state["session_id"] = session_id
        session_state["start_time"] = start_time
        session_state["session_data"] = session_data
        session_state["initialized"] = True
        
        return f"Session initialized successfully!\nArea: {selected_area}\nModel: {model_used}\nSession ID: {session_id}"
    
    def chat(message, history):
        if not session_state["initialized"]:
            return history + [["Please initialize your session first by filling in your details and selecting an inquiry area.", None]]
        
        # Process the query using your existing generate_response function
        response, cost, usage_info, response_time, unsafe_input_flag, top_source_docs = generate_response(
            session_state["session_data"]["employee_id"],
            session_state["session_data"]["employee_name"],
            message,
            session_state["active_chain"],
            zero_shot_classifier,
            toxicity_classifier,
            token_usage_callback
        )
        
        # Log the interaction
        log_interaction(
            session_state["session_data"],
            message,
            response,
            cost,
            usage_info,
            response_time,
            unsafe_input_flag,
            session_state["model_used"],
            top_source_docs
        )
        
        # Return the message and response in the format expected by Gradio Chatbot
        return history + [[message, response]]

    
    def end_session():
        if not session_state["initialized"]:
            return "No active session to end."
        
        # Finalize session data
        end_time = time.time()
        session_state["session_data"]["session_length"] = end_time - session_state["start_time"]
        
        if session_state["session_data"]["total_queries"] > 0:
            session_state["session_data"]["average_response_time"] = (
                session_state["session_data"]["total_response_time"] / 
                session_state["session_data"]["total_queries"]
            )
            session_state["session_data"]["error_rate"] = (
                session_state["session_data"]["error_count"] / 
                session_state["session_data"]["total_queries"]
            )
            session_state["session_data"]["unsafe_input_percentage"] = (
                session_state["session_data"]["unsafe_inputs"] / 
                session_state["session_data"]["total_queries"]
            )
        else:
            session_state["session_data"]["average_response_time"] = 0
            session_state["session_data"]["error_rate"] = 0
            session_state["session_data"]["unsafe_input_percentage"] = 0
        
        # Log the session
        log_session(session_state["session_data"])
        
        # Reset session state
        session_id = session_state["session_id"]
        session_state["initialized"] = False
        
        return f"Session {session_id} ended and logged successfully."
    
    # Create the Gradio interface
    with gr.Blocks(title="Insurellm Knowledge Assistant") as demo:
        gr.Markdown("# Insurellm Knowledge Assistant")
        gr.Markdown("### Please enter your details to start a session")
        
        with gr.Row():
            with gr.Column(scale=1):
                employee_id = gr.Textbox(label="Employee ID", placeholder="Enter your Employee ID")
                employee_name = gr.Textbox(label="Name", placeholder="Enter your full name")
                email = gr.Textbox(label="Email", placeholder="Enter your email address")
                inquiry_choice = gr.Radio(
                    ["Products/Employees", "Contracts/Company"], 
                    label="Area of Inquiry"
                )
                init_button = gr.Button("Initialize Session")
                init_output = gr.Textbox(label="Initialization Status")
                
                init_button.click(
                    initialize_session, 
                    inputs=[employee_id, employee_name, email, inquiry_choice], 
                    outputs=init_output
                )
                
                end_button = gr.Button("End Session")
                end_output = gr.Textbox(label="Session End Status")
                end_button.click(end_session, inputs=[], outputs=end_output)
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(label="Your Question", placeholder="Ask a question about Insurellm...")
                clear = gr.Button("Clear Chat")
                
                msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot])
                clear.click(lambda: None, None, chatbot, queue=False)
        
        gr.Markdown("### Instructions")
        gr.Markdown("""
        1. Enter your Employee ID, Name, and Email
        2. Select your area of inquiry
        3. Click 'Initialize Session' to start
        4. Ask your questions in the chat
        5. Click 'End Session' when you're finished to log your session
        """)
    
    return demo

# --------------------------------------------------------------------------
# 10. Main Entry Point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: Environment setup
    initialize_environment()
    logger = setup_logger()

    # Your desired model paths / parameters
    EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "./models/embeddings/Embedding_Models")
    ZERO_SHOT_MODEL_DIR = os.getenv("ZERO_SHOT_MODEL_DIR", "./models/zero_shot/Deep_Models")
    TOXICITY_MODEL_DIR = os.getenv("TOXICITY_MODEL_DIR", "./models/toxicity/Unitary_Models")
    CROSS_ENCODER_DIR = os.getenv("CROSS_ENCODER_DIR", "./models/cross_encoder/Rerank")

    # EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "D:/Embedding_Models")
    # ZERO_SHOT_MODEL_DIR = "D:/Deep_Models"
    # TOXICITY_MODEL_DIR = "D:/Unitary_Models"
    # CROSS_ENCODER_DIR = "D:/Deep_Models/Rerank"

    # Vector store config
    db_name = "vector_db"

    # LLM model config
    MODEL_GPT4 = "gpt-4o-mini"       # hypothetical GPT-4 local or fine-tuned
    MODEL_GPT35 = "gpt-3.5-turbo"    # typical naming for GPT-3.5
    TEMPERATURE = 0

    # Step 2: Load data
    folders = glob.glob("knowledge-base/*")
    all_docs = []
    for folder in folders:
        folder_docs = load_documents_from_folder(folder)
        all_docs.extend(folder_docs)

    # Step 3: Chunk documents
    chunks = chunk_documents(all_docs)

    print(f"Total number of chunks: {len(chunks)}")
    doc_types = set(doc.metadata['doc_type'] for doc in all_docs)
    print(f"Document types found: {doc_types}")

    # Step 4: Embeddings and vectorstore
    embeddings = get_embeddings(EMBEDDINGS_MODEL_NAME)
    vectorstore = build_vector_store(chunks, embeddings, db_name)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")

    # Step 5: Reranking components
    tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_DIR)
    cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_DIR)

    # (a) Reranker for Products/Employees
    rerank_retriever_products_employees = create_filtered_rerank_retriever(
        vectorstore, tokenizer, cross_encoder_model,
        categories=["products", "employees"]
    )

    # (b) Reranker for Contracts/Company
    rerank_retriever_contracts_company = create_filtered_rerank_retriever(
        vectorstore, tokenizer, cross_encoder_model,
        categories=["contracts", "company"]
    )

    # Step 6: Callback and LLM
    callback_manager = create_callback_manager()
    token_usage_callback = callback_manager.handlers[0]  # We know index 0 is our TokenUsageCallback

    # LLM for Products/Employees
    llm_gpt4 = initialize_llm(MODEL_GPT4, callback_manager, TEMPERATURE)
    # LLM for Contracts/Company
    llm_gpt35 = initialize_llm(MODEL_GPT35, callback_manager, TEMPERATURE)

    # Step 7: Memory and conversation chains
    memory_products_employees = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    conversation_chain_products_employees = create_conversation_chain(
        llm_gpt4, rerank_retriever_products_employees, memory_products_employees
    )

    memory_contracts_company = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    conversation_chain_contracts_company = create_conversation_chain(
        llm_gpt35, rerank_retriever_contracts_company, memory_contracts_company
    )

    # Step 8: Classification pipelines
    zero_shot_classifier = load_zero_shot_classifier(ZERO_SHOT_MODEL_DIR)
    toxicity_classifier = load_toxicity_classifier(TOXICITY_MODEL_DIR)

    # Step 9: Start interactive session (with an additional prompt to select the pipeline)
    
    demo = create_gradio_interface(
        logger=logger,
        conversation_chain_products_employees=conversation_chain_products_employees,
        conversation_chain_contracts_company=conversation_chain_contracts_company,
        zero_shot_classifier=zero_shot_classifier,
        toxicity_classifier=toxicity_classifier,
        token_usage_callback=token_usage_callback
    )

    # Launch the interface
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
