#!/usr/bin/env python3
"""
LangGraph RAG System
A complete RAG (Retrieval-Augmented Generation) system built with LangGraph and LangChain.
"""

import os
import getpass
import nest_asyncio
import tiktoken
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langgraph.graph import START, StateGraph
from IPython.display import Markdown, display

# Apply nest_asyncio for async support in Jupyter
nest_asyncio.apply()

# Set environment variables
def setup_environment():
    """Set up the OpenAI API key."""
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# Define the State for our LangGraph
class State(TypedDict):
    question: str
    context: list[Document]
    response: str

def tiktoken_len(text):
    """Calculate the number of tokens in a text using tiktoken."""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)

def load_and_process_documents():
    """Load and process documents from the data directory."""
    print("Loading documents...")
    directory_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    loan_knowledge_resources = directory_loader.load()
    
    print(f"Loaded {len(loan_knowledge_resources)} documents")
    print(f"Sample content: {loan_knowledge_resources[0].page_content[:200]}...")
    
    return loan_knowledge_resources

def chunk_documents(documents):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=0,
        length_function=tiktoken_len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    return chunks

def setup_vector_store(chunks):
    """Set up Qdrant vector store and add documents."""
    print("Setting up vector store...")
    
    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_dim = 1536  # Dimension for text-embedding-3-small
    
    # Create Qdrant client and collection
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="loan_knowledge_index",
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    
    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="loan_knowledge_index",
        embedding=embedding_model,
    )
    
    # Add documents to vector store
    print("Adding documents to vector store...")
    _ = vector_store.add_documents(documents=chunks)
    
    return vector_store

def create_retriever(vector_store):
    """Create a retriever from the vector store."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever

def create_generator_chain():
    """Create the generator chain with prompt template and LLM."""
    # Create chat prompt template
    HUMAN_TEMPLATE = """
#CONTEXT:
{context}

QUERY:
{query}

Use the provide context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context response with "I don't know"
"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("human", HUMAN_TEMPLATE)
    ])
    
    # Create LLM
    openai_chat_model = ChatOpenAI(model="gpt-4.1-nano")
    
    # Create the complete chain
    generator_chain = chat_prompt | openai_chat_model | StrOutputParser()
    
    return generator_chain

def create_retrieve_node(retriever):
    """Create the retrieve node for the LangGraph."""
    def retrieve(state: State) -> State:
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}
    
    return retrieve

def create_generate_node():
    """Create the generate node for the LangGraph."""
    def generate(state: State) -> State:
        generator_chain = create_generator_chain()
        response = generator_chain.invoke({
            "query": state["question"], 
            "context": state["context"]
        })
        return {"response": response}
    
    return generate

def build_rag_graph(retriever):
    """Build the complete RAG graph."""
    print("Building RAG graph...")
    
    # Create nodes
    retrieve_node = create_retrieve_node(retriever)
    generate_node = create_generate_node()
    
    # Build the graph
    graph_builder = StateGraph(State)
    graph_builder = graph_builder.add_sequence([retrieve_node, generate_node])
    graph_builder.add_edge(START, "retrieve")
    
    # Compile the graph
    graph = graph_builder.compile()
    
    print("Graph built successfully!")
    return graph

def test_rag_system(graph):
    """Test the RAG system with sample queries."""
    test_queries = [
        "Is applying for and securing a student loan in 2025 a terrible idea?",
        "How much loan money can I actually get from the government to go to school these days? Is there a cap?",
        "What grants and scholarships are available for free?",
        "Who is Batman?"  # This should return "I don't know"
    ]
    
    print("\n" + "="*50)
    print("TESTING RAG SYSTEM")
    print("="*50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {query}")
        
        response = graph.invoke({"question": query})
        
        print(f"Response: {response['response']}")
        print("-" * 40)

def main():
    """Main function to run the complete RAG system."""
    print("Setting up LangGraph RAG System...")
    
    # Set up environment
    setup_environment()
    
    # Load and process documents
    documents = load_and_process_documents()
    chunks = chunk_documents(documents)
    
    # Set up vector store and retriever
    vector_store = setup_vector_store(chunks)
    retriever = create_retriever(vector_store)
    
    # Test retriever
    print("\nTesting retriever...")
    test_docs = retriever.invoke("What is the loan repayment period?")
    print(f"Retrieved {len(test_docs)} documents")
    
    # Build and test the graph
    graph = build_rag_graph(retriever)
    test_rag_system(graph)
    
    print("\nRAG system setup complete!")

if __name__ == "__main__":
    main() 