#!/usr/bin/env python3
"""
LangSmith and Evaluation System
A complete RAG system with LangSmith integration and evaluation capabilities.
"""

import os
import getpass
import nest_asyncio
import tiktoken
import pandas as pd
from uuid import uuid4
from datetime import datetime
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langchain.callbacks import LangChainTracer
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate

# Apply nest_asyncio for async support
nest_asyncio.apply()

def setup_environment():
    """Set up environment variables for OpenAI and LangSmith."""
    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key:")
    
    # Set up LangSmith
    unique_id = uuid4().hex[0:8]
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = f"LangSmith - {unique_id}"
    
    # Set LangSmith API key
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass('Enter your LangSmith API key: ')

def load_and_process_documents():
    """Load and process documents from the data directory."""
    print("Loading documents...")
    directory_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    loan_knowledge_resources = directory_loader.load()
    
    print(f"Loaded {len(loan_knowledge_resources)} documents")
    return loan_knowledge_resources

def chunk_documents(documents):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    print("Chunking documents...")
    
    def tiktoken_len(text):
        tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
        return len(tokens)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        length_function=tiktoken_len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Verify chunk sizes
    max_chunk_length = 0
    for chunk in chunks:
        max_chunk_length = max(max_chunk_length, tiktoken_len(chunk.page_content))
    
    print(f"Maximum chunk length: {max_chunk_length} tokens")
    return chunks

def setup_vector_store(chunks):
    """Set up Qdrant vector store and add documents."""
    print("Setting up vector store...")
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    qdrant_vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding=embedding_model,
        location=":memory:"
    )
    
    return qdrant_vectorstore

def create_rag_graph(vector_store):
    """Create the RAG graph with retrieve and generate nodes."""
    print("Creating RAG graph...")
    
    # Create retriever
    retriever = vector_store.as_retriever()
    
    # Create prompt template
    HUMAN_TEMPLATE = """
#CONTEXT:
{context}

QUERY:
{query}

Use the provide context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with "I don't know". Do not make up answers or use external knowledge.
"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", HUMAN_TEMPLATE)
    ])
    
    # Create LLM
    openai_chat_model = ChatOpenAI(model="gpt-4.1-nano")
    
    # Define State
    class State(TypedDict):
        question: str
        context: list[Document]
        response: str
    
    # Create nodes
    def retrieve(state: State) -> State:
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}
    
    def generate(state: State) -> State:
        generator_chain = chat_prompt | openai_chat_model | StrOutputParser()
        response = generator_chain.invoke({
            "query": state["question"], 
            "context": state["context"]
        })
        return {"response": response}
    
    # Build graph
    graph_builder = StateGraph(State)
    graph_builder = graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    rag_graph = graph_builder.compile()
    
    print("RAG graph created successfully!")
    return rag_graph

def tracing_sanity_check(project_name=None):
    """Run a sanity check for LangSmith tracing."""
    print("🔍 Running LangSmith Tracing Sanity Check...")

    # 1. Environment Variables
    print("\n🔐 Environment Variables:")
    api_key = os.getenv("LANGSMITH_API_KEY", "<MISSING>")
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", "<default (https://api.smith.langchain.com)>")
    print(f"  LANGSMITH_API_KEY set: {'✅' if api_key != '<MISSING>' else '❌ MISSING'}")
    print(f"  LANGCHAIN_ENDPOINT: {endpoint}")

    # 2. LangSmith Client Check
    print("\n🌐 LangSmith Client Connection:")
    try:
        client = Client()
        datasets = list(client.list_datasets())
        print(f"  ✅ Connected to LangSmith. {len(datasets)} datasets accessible.")
    except Exception as e:
        print(f"  ❌ Failed to connect to LangSmith Client: {e}")
        return

    # 3. Tracer Initialization
    if not project_name:
        project_name = f"sanity-check-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"\n🧪 Creating tracer for project: {project_name}")
    try:
        tracer = LangChainTracer(project_name=project_name)
        print("  ✅ Tracer created")
    except Exception as e:
        print(f"  ❌ Failed to create tracer: {e}")
        return

    # 4. Runnable trace test
    print("\n⚙️ Testing traceable Runnable...")
    try:
        def dummy_fn(input):
            return {"output": input["input"].upper()}

        dummy = RunnableLambda(dummy_fn)
        result = dummy.invoke({"input": "test"}, config={"callbacks": [tracer]})
        print("  ✅ Dummy Runnable executed successfully")
        print(f"  📦 Output: {result['output']}")
        print("  📦 Check your LangSmith dashboard for a new project titled:", project_name)
    except Exception as e:
        print(f"  ❌ Runnable trace test failed: {e}")

def test_rag_with_tracing(rag_graph):
    """Test the RAG system with LangSmith tracing."""
    print("\nTesting RAG system with tracing...")
    
    tracer = LangChainTracer(project_name=os.environ["LANGSMITH_PROJECT"])
    config = {
        "tags": ["Demo Run"],
        "callbacks": [tracer]
    }
    
    # Test queries
    test_queries = [
        "What is the maximum loan amount I can get from the government to go to school these days?",
        "Is applying for and securing a student loan in 2025 a terrible idea?",
        "What is the airspeed velocity of an unladen swallow?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {query}")
        
        response = rag_graph.invoke({"question": query}, config)
        print(f"Response: {response['response']}")
        
        # Show context for first query
        if i == 1:
            print("\nContext used:")
            for j, context in enumerate(response["context"], 1):
                print(f"{j}. {context.page_content[:100]}...")
        
        print("-" * 40)

def setup_evaluation_dataset():
    """Set up the evaluation dataset in LangSmith."""
    print("\nSetting up evaluation dataset...")
    
    # Clone the data repository if not exists
    import subprocess
    try:
        subprocess.run(["git", "clone", "https://github.com/AI-Maker-Space/DataRepository.git"], 
                      check=True, capture_output=True)
        print("✅ DataRepository cloned successfully")
    except subprocess.CalledProcessError:
        print("⚠️  DataRepository already exists or clone failed")
    
    # Load test data
    test_df = pd.read_csv("DataRepository/student_loan_rag_test_data.csv")
    print(f"Loaded {len(test_df)} test examples")
    
    # Create dataset in LangSmith
    client = Client()
    dataset_name = "langsmith-student-loan-rag"
    
    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name, 
            description="Student Loan RAG Test Questions"
        )
        print(f"✅ Created dataset: {dataset_name}")
    except Exception as e:
        print(f"⚠️  Dataset may already exist: {e}")
        # Try to get existing dataset
        datasets = list(client.list_datasets())
        dataset = next((d for d in datasets if d.name == dataset_name), None)
    
    if dataset:
        # Add examples to dataset
        for triplet in test_df.iterrows():
            triplet = triplet[1]
            try:
                client.create_example(
                    inputs={"question": triplet["question"], "context": triplet["context"]},
                    outputs={"answer": triplet["answer"]},
                    dataset_id=dataset.id
                )
            except Exception as e:
                print(f"⚠️  Example may already exist: {e}")
        
        print(f"✅ Added {len(test_df)} examples to dataset")
    
    return dataset_name

def create_evaluators():
    """Create custom evaluators for the RAG system."""
    print("\nCreating evaluators...")
    
    # Data preparation functions
    def prepare_data_ref(run, example):
        return {
            "prediction": run.outputs["response"],
            "reference": example.outputs["answer"],
            "input": example.inputs["question"]
        }

    def prepare_data_noref(run, example):
        return {
            "prediction": run.outputs["response"],
            "input": example.inputs["question"]
        }

    def prepare_context_ref(run, example):
        return {
            "prediction": run.outputs["response"],
            "reference": example.inputs["context"],
            "input": example.inputs["question"]
        }
    
    # Create evaluators
    eval_llm = ChatOpenAI(model="gpt-4o-mini", tags=["eval_llm"])
    
    cot_qa_evaluator = LangChainStringEvaluator(
        "cot_qa", 
        config={"llm": eval_llm}, 
        prepare_data=prepare_context_ref
    )
    
    unlabeled_dopeness_evaluator = LangChainStringEvaluator(
        "criteria",
        config={
            "criteria": {
                "dopeness": "Is the answer to the question dope, meaning cool - awesome - and legit?"
            },
            "llm": eval_llm,
        },
        prepare_data=prepare_data_noref
    )
    
    labeled_score_evaluator = LangChainStringEvaluator(
        "labeled_score_string",
        config={
            "criteria": {
                "accuracy": "Is the generated answer the same as the reference answer?"
            },
        },
        prepare_data=prepare_data_ref
    )
    
    return [cot_qa_evaluator, unlabeled_dopeness_evaluator, labeled_score_evaluator]

def run_evaluation(rag_graph, dataset_name):
    """Run evaluation on the RAG system."""
    print("\nRunning evaluation...")
    
    evaluators = create_evaluators()
    
    try:
        results = evaluate(
            rag_graph.invoke,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix="Base RAG Evaluation"
        )
        
        print("✅ Evaluation completed successfully!")
        print(f"📊 View results at: {results}")
        
        return results
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def main():
    """Main function to run the complete LangSmith evaluation system."""
    print("Setting up LangSmith Evaluation System...")
    
    # Set up environment
    setup_environment()
    
    # Run tracing sanity check
    tracing_sanity_check()
    
    # Load and process documents
    documents = load_and_process_documents()
    chunks = chunk_documents(documents)
    
    # Set up vector store and RAG graph
    vector_store = setup_vector_store(chunks)
    rag_graph = create_rag_graph(vector_store)
    
    # Test RAG with tracing
    test_rag_with_tracing(rag_graph)
    
    # Set up evaluation dataset
    dataset_name = setup_evaluation_dataset()
    
    # Run evaluation
    evaluation_results = run_evaluation(rag_graph, dataset_name)
    
    print("\n🎉 LangSmith evaluation system setup complete!")
    print("\nNext steps:")
    print("1. Check your LangSmith dashboard for traces and evaluation results")
    print("2. Analyze the evaluation metrics to understand system performance")
    print("3. Iterate on the RAG system based on evaluation feedback")

if __name__ == "__main__":
    main() 