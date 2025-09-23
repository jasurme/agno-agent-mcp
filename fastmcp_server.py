#!/usr/bin/env python3
"""
FastMCP Server for OpenSearch Search Tools
Provides a FastMCP server that exposes OpenSearch search capabilities.
"""

from fastmcp import FastMCP
from mcp_opensearch_tool import OpenSearchMCPTool
import asyncio
import json


search_tool = OpenSearchMCPTool()


mcp = FastMCP("opensearch-search")

@mcp.tool()
def bm25_search(query: str, size: int = 5) -> str:
    """
    Perform BM25 text search on arXiv papers. Best for exact keyword matching.
    
    Args:
        query: Search query text
        size: Number of results to return (default: 5)
    
    Returns:
        JSON string of search results
    """
    try:
        results = search_tool.bm25_search(query, size)
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def vector_search(query: str, size: int = 5) -> str:
    """
    Perform semantic vector search on arXiv papers. Best for understanding meaning and context.
    
    Args:
        query: Search query text
        size: Number of results to return (default: 5)
    
    Returns:
        JSON string of search results
    """
    try:
        results = search_tool.vector_search(query, size)
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def hybrid_search(query: str, size: int = 5, bm25_weight: float = 0.3, vector_weight: float = 0.7) -> str:
    """
    Perform hybrid search combining BM25 and vector search for optimal results.
    
    Args:
        query: Search query text
        size: Number of results to return (default: 5)
        bm25_weight: Weight for BM25 search (default: 0.3)
        vector_weight: Weight for vector search (default: 0.7)
    
    Returns:
        JSON string of search results
    """
    try:
        results = search_tool.hybrid_search(query, size, bm25_weight, vector_weight)
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def get_paper_details(paper_id: str) -> str:
    """
    Get detailed information about a specific paper by paper ID.
    
    Args:
        paper_id: Paper ID of the paper (e.g., 'paper1')
    
    Returns:
        Detailed paper information
    """
    try:
        paper = search_tool.get_paper_details(paper_id)
        
        response = f"Paper Details for {paper_id}:\n\n"
        response += f"Full Text:\n{paper['full_text']}\n"
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
