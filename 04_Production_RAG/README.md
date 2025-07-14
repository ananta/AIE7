<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Production RAG with LangGraph and LangChain</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 4: Pre-Work](https://www.notion.so/Session-4-Production-Grade-RAG-with-LangChain-224cd547af3d8092a8a8faa917b5417b?source=copy_link#224cd547af3d8079a747e295b73cbcdd)| [Session 4: Production-Grade RAG with LangChain and LangSmith](https://www.notion.so/Session-4-Production-Grade-RAG-with-LangChain-and-LangSmith-224cd547af3d8092a8a8faa917b5417b) | [Recording!](https://us02web.zoom.us/rec/share/ZVh_CHPQnhYd-kYEwyMF2wG-QHDTPku8cQiGV752YtCFXine2KhtbvDLszMqDPBv.z_vdYbqqEuMhHOH5)  (%PP12Qj$) | [Session 4 Slides](https://www.canva.com/design/DAGkR5kF6Hk/AUdlJOngdbF-ETsp67TdQA/edit?utm_content=DAGkR5kF6Hk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 4 Assignment: Production RAG](https://forms.gle/wdKASjYbDRrsht3N8) | [AIE7 Feedback 7/3](https://forms.gle/YSgU6V9GqBhWXLXw8)


# Build 🏗️

If running locally:

1. `uv sync`
2. Open the notebook
3. Select the venv created by `uv sync` as your kernel

Run the notebook and complete the contained tasks:

- 🤝 Breakout Room #1:
    1. Install LangGraph
    2. Understanding States and Nodes
    3. Building a Basic Graph
    4. Implementing a Simple RAG Graph
    5. Extending the Graph with Complex Flows

Next, run the LangSmith and Evaluation notebook and complete the contained tasks:

- 🤝 Breakout Room #2:
    1. Dependencies and OpenAI API Key
    2. LangGraph RAG
    3. Setting Up LangSmith
    4. Examining the Trace in LangSmith!
    5. Create Testing Dataset
    6. Evaluation

# Ship 🚢

- The completed notebook. 
- 5min. Loom Video

# Share 🚀
- Walk through your notebook and explain what you've completed in the Loom video
- Make a social media post about your final application and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned

# Submitting Your Homework

Follow these steps to prepare and submit your homework:
1. Create a branch of your `AIE7` repo to track your changes. Example command: `git checkout -b s04-assignment`
2. Responding to the activities and questions in both the `Assignment_Introduction_to_LCEL_and_LangGraph_LangChain_Powered_RAG.ipynb`and `LangSmith_and_Evaluation` notebooks:
    + Option 1: Provide your responses in a separate markdown document:
      + Create a markdown document in the `04_Production_RAG` folder of your assignment branch (for example “ACTIVITIES_QUESTIONS.md”):
      + Copy the activities and questions into the document
      + Provide your responses to these activities and questions
    + Option 2: Respond to the activities and questions inline in the notebooks:
      + Edit the markdown cells of the activities and questions then enter your responses
      + NOTE: Remember to create a header (example: `##### ✅ Answer:`) to help the grader find your responses
3. Add (if you created a separate document), commit, and push your responses to your `origin` repository.

> _NOTE on the `Assignment_Introduction_to_LCEL_and_LangGraph_LangChain_Powered_RAG` notebook: You will also need to enter your response to Question #1 in the code cell directly below it which contains this line of code:_
    ```
    embedding_dim =  # YOUR ANSWER HERE
    ```

# LangGraph RAG System

A complete RAG (Retrieval-Augmented Generation) system built with LangGraph and LangChain for answering questions about student loan documents.

## Features

- **Document Loading**: Loads PDF documents from the `data/` directory
- **Text Chunking**: Splits documents into manageable chunks using RecursiveCharacterTextSplitter
- **Vector Embeddings**: Uses OpenAI's text-embedding-3-small model for document embeddings
- **Vector Database**: Stores embeddings in Qdrant vector database
- **LangGraph Pipeline**: Implements a two-node graph (retrieve → generate) for RAG
- **OpenAI Integration**: Uses GPT-4.1-nano for response generation

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API Key**:
   - You'll be prompted to enter your OpenAI API key when running the script
   - Or set it as an environment variable: `export OPENAI_API_KEY="your-key-here"`

3. **Prepare Data**:
   - Place your PDF documents in the `data/` directory
   - The system will automatically load all PDF files from this directory

## Usage

### Running the Complete System

```bash
python langgraph_rag_system.py
```

This will:
1. Load and process all PDF documents from the `data/` directory
2. Chunk the documents into smaller pieces
3. Create embeddings and store them in a Qdrant vector database
4. Build a LangGraph with retrieve and generate nodes
5. Test the system with sample queries

### Using the System Programmatically

```python
from langgraph_rag_system import build_rag_graph, load_and_process_documents, chunk_documents, setup_vector_store

# Load and process documents
documents = load_and_process_documents()
chunks = chunk_documents(documents)

# Set up vector store and retriever
vector_store = setup_vector_store(chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Build the graph
graph = build_rag_graph(retriever)

# Query the system
response = graph.invoke({"question": "What is the maximum loan amount for students?"})
print(response["response"])
```

## System Architecture

### State Definition
```python
class State(TypedDict):
    question: str
    context: list[Document]
    response: str
```

### Graph Nodes

1. **Retrieve Node**: 
   - Takes a question from the state
   - Retrieves relevant documents using the vector retriever
   - Updates the state with retrieved context

2. **Generate Node**:
   - Takes the question and context from the state
   - Generates a response using the LLM chain
   - Updates the state with the final response

### Graph Flow
```
START → retrieve → generate → END
```

## Configuration

### Embedding Model
- Model: `text-embedding-3-small`
- Dimension: 1536
- Distance Metric: Cosine

### Text Chunking
- Chunk Size: 750 tokens
- Chunk Overlap: 0
- Tokenizer: tiktoken (GPT-4o)

### LLM
- Model: `gpt-4.1-nano`
- Temperature: Default
- Max Tokens: Default

### Vector Retrieval
- Number of documents retrieved: 5
- Distance metric: Cosine similarity

## Customization

### Adding New Document Types
To support different document types, modify the `load_and_process_documents()` function:

```python
from langchain_community.document_loaders import CSVLoader, TextLoader

def load_and_process_documents():
    # Load PDFs
    pdf_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    pdf_docs = pdf_loader.load()
    
    # Load CSVs
    csv_loader = DirectoryLoader("data", glob="**/*.csv", loader_cls=CSVLoader)
    csv_docs = csv_loader.load()
    
    return pdf_docs + csv_docs
```

### Modifying the Prompt Template
To change how the system generates responses, modify the `HUMAN_TEMPLATE` in `create_generator_chain()`:

```python
HUMAN_TEMPLATE = """
Based on the following context, answer the user's question:

Context: {context}
Question: {query}

Provide a clear and accurate answer based only on the provided context.
"""
```

### Adding New Graph Nodes
To extend the graph with additional processing steps:

```python
def create_validate_node():
    def validate(state: State) -> State:
        # Add validation logic here
        return state
    return validate

# Add to graph
graph_builder = graph_builder.add_sequence([retrieve, validate, generate])
```

## Testing

The system includes built-in tests with sample queries:
- Student loan questions (should return relevant answers)
- Out-of-domain questions (should return "I don't know")

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**: Make sure your API key is valid and has sufficient credits
2. **Memory Issues**: For large document collections, consider using a persistent Qdrant instance
3. **Import Errors**: Ensure all dependencies are installed with the correct versions

### Performance Optimization

- Use a persistent Qdrant instance for production
- Implement caching for frequently asked questions
- Consider using a more powerful LLM for complex queries
- Implement parallel processing for large document collections

## License

This project is for educational purposes. Please ensure you comply with OpenAI's usage policies and terms of service.
