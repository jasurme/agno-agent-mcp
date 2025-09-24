# MCP Demo


## 🚀 Quick Start

### Prerequisites

Before running this project, ensure you have the following installed:
You are expected to have
- Docker installed

- python 3.11
and git


### 2. Set Up Environment Variables
1. create virtual environment to install libraries

python3.11 -m venv env11 
if you are on MacOS: to activate: source env11/bin/activate
on Windows: env11\Scripts\activate


2. Create `.env` . We need OpenAI api key. 
- (no need to change litellm key. it is what we set ourselves inside our system)
```bash

OPENAI_API_KEY= actual_api_key
LITELLM_MASTER_KEY=demo-key-123
HUGGINGFACE_API_KEY=HF_key
```


### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Docker Services

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
python indexer.py
```

This will:
- read files from papers/pdfs
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

if you open litellm_config.yaml, HF model is taken in comments. 
if you want to use HF, just remove comments and make Openai model and api key lines in comments and change model name in agno_agent.py

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: "your-openai-api-key"
```

Advice is to decide one and use it. if you use one and switch to another, after changing litellm_config.yaml, restart litellm (in terminal): 

docker-compose restart litellm-proxy


## 📁 Project Structure

```
mcp-demo/
├── agno_agent.py              # Main agent script
├── indexer.py                 # PDF processing and indexing script
├── fastmcp_server.py          # FastMCP server
├── mcp_opensearch_tool.py     # OpenSearch MCP tool
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

## 🔄 Adding New PDFs

To add new PDFs to the system:

1. **Place PDFs in the papers directory:**
   ```bash
   mkdir -p papers/pdfs
   # Copy your PDFs to papers/pdfs/
   ```

2. **Run the indexer:**
   ```bash
   python3 indexer.py
   ```

3. **Run the agent:**
   ```bash
   python3 agno_agent.py
   ```

The indexer will automatically:
- Read PDFs from `papers/pdfs/`
- Extract text from PDFs
- Split into chunks
- Generate embeddings
- Index in OpenSearch

## 🚀 Advanced Usage


### Adding New PDFs

1. Place PDF files in the `papers/` directory
2. Run the indexer: `python opensearch_indexer.py`
3. The new documents will be automatically processed and indexed

## 🔄 Stopping the Services

To stop all services:

```bash
docker-compose down
```

To stop and remove all data:

```bash
docker-compose down -v
```
