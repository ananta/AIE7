# LangSmith and Evaluation System

A complete RAG (Retrieval-Augmented Generation) system with LangSmith integration and evaluation capabilities for monitoring, testing, debugging, and evaluating LangChain applications.

## Features

- **RAG System**: Complete RAG pipeline using LangGraph
- **LangSmith Integration**: Full tracing and monitoring capabilities
- **Evaluation Framework**: Multiple evaluators for system performance
- **Document Processing**: PDF loading, chunking, and vector storage
- **Custom Evaluators**: Chain-of-thought QA, accuracy, and "dopeness" evaluators

## Prerequisites

1. **OpenAI API Key**: For embeddings and LLM generation
2. **LangSmith Account**: Sign up at [LangSmith](https://www.langchain.com/langsmith)
3. **LangSmith API Key**: Get from LangSmith Settings → API Keys

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your PDF documents in the `data/` directory. The system will automatically load all PDF files.

### 3. Set Up API Keys

The system will prompt you for:
- OpenAI API Key
- LangSmith API Key

## Usage

### Running the Complete System

```bash
python langsmith_evaluation_system.py
```

This will:
1. Set up environment variables
2. Run LangSmith tracing sanity check
3. Load and process PDF documents
4. Create vector embeddings and store in Qdrant
5. Build RAG graph with retrieve and generate nodes
6. Test the system with sample queries
7. Set up evaluation dataset in LangSmith
8. Run comprehensive evaluation

### System Components

#### 1. Document Processing
```python
# Load documents
documents = load_and_process_documents()

# Chunk documents (200 tokens max)
chunks = chunk_documents(documents)

# Set up vector store
vector_store = setup_vector_store(chunks)
```

#### 2. RAG Graph Creation
```python
# Create RAG graph
rag_graph = create_rag_graph(vector_store)

# Test with tracing
test_rag_with_tracing(rag_graph)
```

#### 3. LangSmith Tracing
```python
# Set up tracing
tracer = LangChainTracer(project_name=os.environ["LANGSMITH_PROJECT"])
config = {"callbacks": [tracer]}

# Run with tracing
response = rag_graph.invoke({"question": "Your question"}, config)
```

#### 4. Evaluation Setup
```python
# Set up evaluation dataset
dataset_name = setup_evaluation_dataset()

# Run evaluation
results = run_evaluation(rag_graph, dataset_name)
```

## LangSmith Integration

### Tracing Sanity Check

The system includes a comprehensive tracing sanity check that verifies:

1. **Environment Variables**: Checks if LangSmith API key is set
2. **Client Connection**: Tests connection to LangSmith API
3. **Tracer Creation**: Verifies tracer initialization
4. **Runnable Testing**: Tests traceable function execution

### Evaluation Metrics

The system uses three types of evaluators:

#### 1. Chain-of-Thought QA Evaluator
- **Purpose**: Evaluates answer quality using context
- **Method**: Uses GPT-4o-mini to assess response quality
- **Input**: Question, context, and generated response

#### 2. "Dopeness" Evaluator
- **Purpose**: Assesses if answers are "cool, awesome, and legit"
- **Method**: Custom criteria evaluation
- **Input**: Question and generated response

#### 3. Accuracy Evaluator
- **Purpose**: Compares generated answers to reference answers
- **Method**: String comparison with labeled data
- **Input**: Question, generated response, and reference answer

## Configuration

### RAG System Settings

```python
# Chunking settings
chunk_size = 200  # tokens
chunk_overlap = 0

# Embedding model
embedding_model = "text-embedding-3-small"

# LLM model
llm_model = "gpt-4.1-nano"

# Evaluation LLM
eval_llm = "gpt-4o-mini"
```

### LangSmith Settings

```python
# Environment variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = f"LangSmith - {unique_id}"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
```

## Evaluation Dataset

The system uses a student loan RAG test dataset with:
- Questions about student loans
- Context from loan documents
- Reference answers for comparison

### Dataset Structure
```csv
question,context,answer
"What is the maximum loan amount?", "Context about loan limits...", "Reference answer..."
```

## Monitoring and Debugging

### LangSmith Dashboard

1. **Traces**: View detailed execution traces
2. **Projects**: Organize experiments by project
3. **Datasets**: Manage test datasets
4. **Evaluations**: View evaluation results and metrics

### Key Metrics to Monitor

1. **Retrieval Quality**: Are relevant documents being retrieved?
2. **Generation Quality**: Are responses accurate and helpful?
3. **Response Time**: How fast is the system?
4. **Token Usage**: Cost optimization

## Customization

### Adding New Evaluators

```python
def create_custom_evaluator():
    return LangChainStringEvaluator(
        "criteria",
        config={
            "criteria": {
                "custom_metric": "Your evaluation criteria"
            },
            "llm": eval_llm,
        },
        prepare_data=prepare_data_function
    )
```

### Modifying the RAG Pipeline

```python
def create_custom_rag_graph(vector_store):
    # Add custom nodes
    def custom_node(state: State) -> State:
        # Custom processing logic
        return state
    
    # Build graph with custom nodes
    graph_builder = StateGraph(State)
    graph_builder = graph_builder.add_sequence([retrieve, custom_node, generate])
    return graph_builder.compile()
```

### Custom Data Preparation

```python
def prepare_custom_data(run, example):
    return {
        "prediction": run.outputs["response"],
        "reference": example.outputs["custom_reference"],
        "input": example.inputs["custom_input"]
    }
```

## Troubleshooting

### Common Issues

1. **LangSmith Connection Failed**
   - Verify API key is correct
   - Check internet connection
   - Ensure LangSmith account is active

2. **Evaluation Dataset Not Found**
   - Check if DataRepository was cloned successfully
   - Verify CSV file exists in expected location

3. **Memory Issues**
   - Reduce chunk size for large documents
   - Use persistent vector store instead of in-memory

4. **API Rate Limits**
   - Implement retry logic
   - Use batch processing for large datasets

### Performance Optimization

1. **Vector Store**: Use persistent Qdrant for production
2. **Caching**: Implement response caching
3. **Batch Processing**: Process multiple queries together
4. **Model Selection**: Choose appropriate models for cost/performance balance

## Advanced Features

### Custom Tracing

```python
# Add custom tags and metadata
config = {
    "tags": ["production", "v1.0"],
    "metadata": {"user_id": "123", "session_id": "abc"},
    "callbacks": [tracer]
}
```

### Batch Evaluation

```python
# Run evaluation on multiple datasets
datasets = ["dataset1", "dataset2", "dataset3"]
for dataset in datasets:
    results = run_evaluation(rag_graph, dataset)
```

### A/B Testing

```python
# Compare different RAG configurations
config_a = create_rag_graph(vector_store_a)
config_b = create_rag_graph(vector_store_b)

results_a = run_evaluation(config_a, dataset_name)
results_b = run_evaluation(config_b, dataset_name)
```

## Best Practices

1. **Regular Evaluation**: Run evaluations after each system change
2. **Monitor Costs**: Track token usage and API costs
3. **Version Control**: Keep track of system versions and configurations
4. **Documentation**: Document changes and their impact on performance
5. **Security**: Never commit API keys to version control

## License

This project is for educational purposes. Please ensure you comply with OpenAI's and LangSmith's usage policies and terms of service. 