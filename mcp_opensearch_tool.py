#!/usr/bin/env python3
"""
MCP OpenSearch Tool
Provides search capabilities for the arXiv papers index using different search techniques.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from mcp import types
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

class OpenSearchMCPTool:
    def __init__(self, opensearch_host: str = "localhost", opensearch_port: int = 9200):
        """Initialize the MCP tool with OpenSearch connection."""
        self.client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=60
        )
        
        self.embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        self.index_name = "arxiv_papers"
    
    def bm25_search(self, query: str, size: int = 5) -> List[Dict[str, Any]]:
        """Perform BM25 text search."""
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["full_text"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    "size": size,
                    "_source": ["paper_id", "full_text"]
                }
            )
            return response['hits']['hits']
        except Exception as e:
            raise Exception(f"BM25 search failed: {str(e)}")
    
    def vector_search(self, query: str, size: int = 5) -> List[Dict[str, Any]]:
        """Perform dense vector search using semantic embeddings."""
        try:
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "nested": {
                            "path": "chunks",
                            "query": {
                                "knn": {
                                    "chunks.embedding": {
                                        "vector": query_embedding,
                                        "k": size
                                    }
                                }
                            }
                        }
                    },
                    "size": size,
                    "_source": ["paper_id", "full_text"]
                }
            )
            return response['hits']['hits']
        except Exception as e:
            raise Exception(f"Vector search failed: {str(e)}")
    
    def hybrid_search(self, query: str, size: int = 5, bm25_weight: float = 0.3, vector_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Perform hybrid search combining BM25 and vector search."""
        try:
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["full_text"],
                                        "type": "best_fields",
                                        "fuzziness": "AUTO",
                                        "boost": bm25_weight
                                    }
                                },
                                {
                                    "nested": {
                                        "path": "chunks",
                                        "query": {
                                            "knn": {
                                                "chunks.embedding": {
                                                    "vector": query_embedding,
                                                    "k": size
                                                }
                                            }
                                        },
                                        "boost": vector_weight
                                    }
                                }
                            ]
                        }
                    },
                    "size": size,
                    "_source": ["paper_id", "full_text"]
                }
            )
            return response['hits']['hits']
        except Exception as e:
            raise Exception(f"Hybrid search failed: {str(e)}")
    
    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific paper."""
        try:
            response = self.client.get(
                index=self.index_name,
                id=paper_id,
                _source=True
            )
            return response['_source']
        except Exception as e:
            raise Exception(f"Failed to get paper details: {str(e)}")
    


search_tool = OpenSearchMCPTool()

server = Server("opensearch-search")

@server.list_tools()
async def list_tools() -> List[types.Tool]:
    """List available search tools."""
    return [
        types.Tool(
            name="bm25_search",
            description="Perform BM25 text search on arXiv papers. Best for exact keyword matching.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "size": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="vector_search",
            description="Perform semantic vector search on arXiv papers. Best for understanding meaning and context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "size": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="hybrid_search",
            description="Perform hybrid search combining BM25 and vector search for optimal results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "size": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    },
                    "bm25_weight": {
                        "type": "number",
                        "description": "Weight for BM25 search (default: 0.3)",
                        "default": 0.3
                    },
                    "vector_weight": {
                        "type": "number",
                        "description": "Weight for vector search (default: 0.7)",
                        "default": 0.7
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_paper_details",
            description="Get detailed information about a specific paper by paper ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "Paper ID of the paper (e.g., 'paper1')"
                    }
                },
                "required": ["paper_id"]
            }
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls."""
    try:
        if name == "bm25_search":
            query = arguments["query"]
            size = arguments.get("size", 5)
            results = search_tool.bm25_search(query, size)
            
            response = f"BM25 Search Results for '{query}':\n\n"
            for i, hit in enumerate(results, 1):
                source = hit['_source']
                response += f"{i}. [{hit['_score']:.2f}] Paper: {source['paper_id']}\n"
                response += f"   Text: {source['full_text'][:300]}...\n\n"
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "vector_search":
            query = arguments["query"]
            size = arguments.get("size", 5)
            results = search_tool.vector_search(query, size)
            
            response = f"Vector Search Results for '{query}':\n\n"
            for i, hit in enumerate(results, 1):
                source = hit['_source']
                response += f"{i}. [{hit['_score']:.2f}] Paper: {source['paper_id']}\n"
                response += f"   Text: {source['full_text'][:300]}...\n\n"
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "hybrid_search":
            query = arguments["query"]
            size = arguments.get("size", 5)
            bm25_weight = arguments.get("bm25_weight", 0.3)
            vector_weight = arguments.get("vector_weight", 0.7)
            results = search_tool.hybrid_search(query, size, bm25_weight, vector_weight)
            
            response = f"Hybrid Search Results for '{query}':\n\n"
            for i, hit in enumerate(results, 1):
                source = hit['_source']
                response += f"{i}. [{hit['_score']:.2f}] Paper: {source['paper_id']}\n"
                response += f"   Text: {source['full_text'][:300]}...\n\n"
            
            return [types.TextContent(type="text", text=response)]
        
        elif name == "get_paper_details":
            paper_id = arguments["paper_id"]
            paper = search_tool.get_paper_details(paper_id)
            
            response = f"Paper Details for {paper_id}:\n\n"
            response += f"Full Text:\n{paper['full_text']}\n"
            
            return [types.TextContent(type="text", text=response)]
        
        
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="opensearch-search",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
