# PDF-based Q&A System using RAG (Retrieval-Augmented Generation)

A powerful question-answering system that uses Retrieval-Augmented Generation (RAG) to answer questions based on PDF documents. The system processes PDF documents, creates embeddings, stores them in a vector database, and provides accurate answers by retrieving relevant context.

## 🚀 Features

- **PDF Document Processing**: Automatically loads and processes PDF documents from a directory
- **Document Chunking**: Splits large documents into manageable chunks for better retrieval
- **Vector Database**: Uses ChromaDB for efficient similarity search
- **Multiple Embedding Options**: Supports both AWS Bedrock and Ollama embeddings
- **Intelligent Querying**: Retrieves the most relevant document chunks for answering questions
- **Local LLM Integration**: Uses Ollama with Llama2 for generating responses
- **Automated Testing**: Includes test cases to validate system responses

## 🏗️ Architecture

The system consists of four main components:

1. **Embedding Function** ([get_embedding_function.py](get_embedding_function.py)): Provides text embedding capabilities using AWS Bedrock Titan
2. **Database Population** ([populate_database.py](populate_database.py)): Processes PDFs and populates the vector database
3. **Query System** ([query_data.py](query_data.py)): Handles user queries and retrieves relevant answers
4. **Testing Framework** ([test_rag.py](test_rag.py)): Validates system responses against expected answers

## 📋 Prerequisites

- Python 3.8+
- AWS Account (for Bedrock embeddings)
- Ollama installed locally
- Required Python packages (see Installation section)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PDF-based-Q-A-System-using-RAG-
   ```

2. **Install required packages**
   ```bash
   pip install langchain-community langchain chromadb pypdf boto3
   ```

3. **Install and set up Ollama**
   ```bash
   # Download and install Ollama from https://ollama.ai/
   # Pull the required model
   ollama pull llama2
   ```

4. **Configure AWS credentials**
   ```bash
   # Set up AWS credentials for Bedrock access
   aws configure
   ```
   or set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   ```

## 📁 Project Structure

```
PDF-based-Q-A-System-using-RAG-/
├── data/                          # Directory for PDF documents
├── chroma/                        # ChromaDB vector database (auto-created)
├── get_embedding_function.py      # Embedding function configuration
├── populate_database.py           # Database population script
├── query_data.py                  # Query processing system
├── test_rag.py                    # Testing framework
└── README.md                      # Project documentation
```

## 🚀 Quick Start

### 1. Prepare Your Documents
Create a `data` directory and place your PDF documents inside:
```bash
mkdir data
# Copy your PDF files to the data directory
```

### 2. Populate the Database
Process your PDF documents and create the vector database:
```bash
python populate_database.py
```

To reset the database and start fresh:
```bash
python populate_database.py --reset
```

### 3. Query the System
Ask questions about your documents:
```bash
python query_data.py "What is the main topic of the document?"
```

### 4. Run Tests
Validate the system with predefined test cases:
```bash
python test_rag.py
```

## 🔧 Configuration

### Embedding Function
The system uses AWS Bedrock Titan embeddings by default. You can modify the embedding configuration in [get_embedding_function.py](get_embedding_function.py):

```python
def get_embedding_function():
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'  # Update with your AWS region
    )
    
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v1"
    )
    
    return embeddings
```

### Document Processing Settings
Adjust chunk size and overlap in [populate_database.py](populate_database.py):

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Size of each text chunk
    chunk_overlap=80,    # Overlap between chunks
    length_function=len,
    is_separator_regex=False,
)
```

### Query Configuration
Modify the number of retrieved documents and prompt template in [query_data.py](query_data.py):

```python
results = db.similarity_search_with_score(query_text, k=5)  # Number of chunks to retrieve
```

## 📊 How It Works

1. **Document Loading**: PDFs are loaded from the `data` directory
2. **Text Splitting**: Documents are split into chunks with overlap for context preservation
3. **Embedding Generation**: Each chunk is converted to vector embeddings using AWS Bedrock
4. **Vector Storage**: Embeddings are stored in ChromaDB with metadata
5. **Query Processing**: User questions are embedded and matched against stored chunks
6. **Context Retrieval**: Most relevant chunks are retrieved based on similarity
7. **Answer Generation**: Llama2 generates answers using the retrieved context

## 🧪 Testing

The system includes automated tests for validation:

- **Monopoly Rules Test**: Tests knowledge extraction from game rule PDFs
- **Ticket to Ride Test**: Validates specific factual information retrieval

Add your own test cases in [test_rag.py](test_rag.py):

```python
def test_your_document():
    assert query_and_validate(
        question="Your question here",
        expected_response="Expected answer",
    )
```

## 🔍 Troubleshooting

### Common Issues

1. **AWS Credentials Error**
   - Ensure AWS credentials are properly configured
   - Check that your AWS region supports Bedrock

2. **Ollama Connection Error**
   - Verify Ollama is running: `ollama serve`
   - Confirm Llama2 model is installed: `ollama pull llama2`

3. **No Documents Found**
   - Check that PDF files are in the `data` directory
   - Ensure PDFs are readable and not corrupted

4. **Empty Responses**
   - Verify the database was populated successfully
   - Check if your question is related to the document content

## 📈 Performance Tips

- Use smaller chunk sizes for more precise answers
- Increase chunk overlap for better context continuity
- Adjust the number of retrieved chunks (`k` parameter) based on your needs
- Consider using more powerful embedding models for better accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for more details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the code documentation
3. Create an issue in the repository

---

**Note**: This system requires active AWS Bedrock and Ollama installations. Make sure both services are properly configured before running the application.