# MCP Demo - OpenSearch with FastMCP and Agno Framework

This project demonstrates a complete MCP (Model Context Protocol) implementation using OpenSearch, FastMCP, and the Agno framework. It includes PDF indexing, multiple search types (BM25, Dense Vector, Hybrid), and LLM integration through LiteLLM proxy.

## 🚀 Quick Start

### Prerequisites

Before running this project, ensure you have the following installed:

1. **Docker & Docker Compose**
   - [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - [Install Docker Compose](https://docs.docker.com/compose/install/)

2. **Python 3.8+**
   - [Download Python](https://www.python.org/downloads/)
   - Verify installation: `python --version`

3. **Git**
   - [Install Git](https://git-scm.com/downloads)

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 5GB free space
- **OS**: macOS, Linux, or Windows with WSL2

## 📋 Installation Steps

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd mcp-demo
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Docker Services

```bash
docker-compose up -d
```

This will start:
- **OpenSearch** (port 9200)
- **OpenSearch Dashboards** (port 5601)
- **Ollama** (port 11434)
- **LiteLLM Proxy** (port 4000)

### 4. Wait for Services to Initialize

Wait approximately 2-3 minutes for all services to fully start up. You can check the status with:

```bash
docker-compose ps
```

All services should show "Up" status.

### 5. Index Sample PDFs

```bash
python opensearch_indexer.py
```

This will:
- Download sample PDFs from arXiv
- Process and chunk the documents
- Generate embeddings using HuggingFace BGE model
- Index everything in OpenSearch

### 6. Run the Demo

```bash
python agno_agent.py
```

## 🎯 What the Demo Does

The demo performs three types of searches on the indexed documents:

1. **BM25 Search** - Keyword-based search
2. **Dense Vector Search** - Semantic search using embeddings
3. **Hybrid Search** - Combines both BM25 and vector search

For each search type, it:
- Shows the first 300 characters of results
- Uses the full hybrid search results as context for the LLM
- Generates an AI response using the LiteLLM proxy

## 🔧 Configuration

### LLM Configuration

The project uses OpenAI's GPT-4o-mini model through LiteLLM proxy. The API key is configured in `litellm_config.yaml`:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: "your-openai-api-key"
```

### OpenSearch Configuration

OpenSearch is configured with:
- **Index**: `arxiv_papers`
- **Embedding Model**: HuggingFace BGE (runs locally)
- **Chunk Size**: 1500 characters with 300 character overlap

## 📁 Project Structure

```
mcp-demo/
├── agno_agent.py              # Main agent script
├── fastmcp_server.py          # FastMCP server
├── mcp_opensearch_tool.py     # OpenSearch MCP tool
├── opensearch_indexer.py      # PDF indexing script
├── pdf_processor.py           # PDF processing utilities
├── docker-compose.yml         # Docker services configuration
├── litellm_config.yaml        # LiteLLM proxy configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🛠️ Troubleshooting

### Common Issues

1. **Docker Services Not Starting**
   ```bash
   # Check Docker is running
   docker --version
   
   # Restart services
   docker-compose down
   docker-compose up -d
   ```

2. **Memory Issues**
   - Ensure you have at least 8GB RAM available
   - Close other memory-intensive applications
   - Increase Docker memory limits in Docker Desktop settings

3. **Port Conflicts**
   - Ensure ports 9200, 5601, 11434, and 4000 are not in use
   - Check with: `netstat -an | grep :9200`

4. **Python Dependencies**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

5. **OpenSearch Connection Issues**
   ```bash
   # Check OpenSearch health
   curl http://localhost:9200/_cluster/health
   ```

### Logs and Debugging

- **Docker logs**: `docker-compose logs [service-name]`
- **OpenSearch logs**: `docker-compose logs opensearch`
- **LiteLLM logs**: `docker-compose logs litellm`

## 🔍 Testing the System

### 1. Test OpenSearch
```bash
curl http://localhost:9200/_cluster/health
```

### 2. Test LiteLLM Proxy
```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 3. Test the Demo
```bash
python agno_agent.py
```

## 📊 Expected Output

When running the demo, you should see:

```
Processing query: 'when was web audio api first introduced?'
============================================================

BM25 Search Results:
------------------------------
  1. [0.85] The Web Audio API was first introduced in 2011 as part of the W3C specification...
  2. [0.82] Browser fingerprinting using Web Audio API has been studied extensively...
  3. [0.78] The API provides real-time audio processing capabilities...

Dense Vector Search Results:
------------------------------
  1. [0.92] Web Audio API implementation details and browser compatibility...
  2. [0.89] Audio fingerprinting techniques using Web Audio API...
  3. [0.87] Historical development of Web Audio standards...

Hybrid Search Results:
------------------------------
  1. [0.94] The Web Audio API was first introduced in 2011...
  2. [0.91] Browser implementations and compatibility issues...
  3. [0.88] Security and privacy considerations...

LLM Response:
----------------------------------------
The Web Audio API was first introduced in 2011 as part of the W3C specification.
```

## 🚀 Advanced Usage

### Custom Queries

You can modify the sample queries in `agno_agent.py`:

```python
sample_queries = [
    "your custom query here",
    "another question about the documents"
]
```

### Adding New PDFs

1. Place PDF files in the `papers/` directory
2. Run the indexer: `python opensearch_indexer.py`
3. The new documents will be automatically processed and indexed

### Modifying Search Parameters

Edit the search methods in `agno_agent.py` to adjust:
- Number of results returned
- BM25/Vector search weights
- Chunk size and overlap

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure Docker services are running properly
4. Check the logs for specific error messages

## 🔄 Stopping the Services

To stop all services:

```bash
docker-compose down
```

To stop and remove all data:

```bash
docker-compose down -v
```

## 📝 License

This project is for demonstration purposes. Please ensure you have appropriate licenses for any commercial use of the included technologies.
