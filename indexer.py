#!/usr/bin/env python3
"""
PDF Indexer for OpenSearch
Downloads PDFs, processes them, and indexes them in OpenSearch with BGE embeddings.
"""

import os
import json
import time
import requests
import PyPDF2
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer
import re

class PDFIndexer:
    def __init__(self, opensearch_host: str = "localhost", opensearch_port: int = 9200):
        """Initialize OpenSearch client and BGE model."""
        self.opensearch_host = opensearch_host
        self.opensearch_port = opensearch_port
        self.index_name = "research_papers"
        
        # Initialize OpenSearch client
        self.client = OpenSearch(
            hosts=[{'host': self.opensearch_host, 'port': self.opensearch_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection
        )
        
        # Initialize BGE model for embeddings
        print("Loading BGE embedding model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        print("BGE model loaded successfully!")
        
        # Create index if it doesn't exist
        self._create_index()
    
    def _create_index(self):
        """Create OpenSearch index with proper mapping."""
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "paper_id": {"type": "keyword"},
                        "title": {"type": "text", "analyzer": "standard"},
                        "authors": {"type": "text", "analyzer": "standard"},
                        "abstract": {"type": "text", "analyzer": "standard"},
                        "full_text": {"type": "text", "analyzer": "standard"},
                        "chunk_text": {"type": "text", "analyzer": "standard"},
                        "chunk_embedding": {
                            "type": "dense_vector",
                            "dims": 1024,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "chunk_index": {"type": "integer"},
                        "total_chunks": {"type": "integer"},
                        "publication_date": {"type": "date"},
                        "url": {"type": "keyword"},
                        "file_path": {"type": "keyword"}
                    }
                }
            }
            
            self.client.indices.create(index=self.index_name, body=mapping)
            print(f"Created index: {self.index_name}")
        else:
            print(f"Index {self.index_name} already exists")
    
    def extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_text_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber."""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods."""
        print(f"Extracting text from: {pdf_path}")
        
        # Try pdfplumber first (usually better)
        text = self.extract_text_with_pdfplumber(pdf_path)
        
        # If pdfplumber fails or returns empty, try PyPDF2
        if not text.strip():
            print("pdfplumber failed, trying PyPDF2...")
            text = self.extract_text_with_pypdf2(pdf_path)
        
        if not text.strip():
            print(f"Warning: Could not extract text from {pdf_path}")
            return ""
        
        # Clean up the text
        text = self.clean_text(text)
        print(f"Extracted {len(text)} characters")
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using BGE model."""
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 1024  # Return zero vector if embedding fails
    
    def index_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Index a single paper in OpenSearch."""
        try:
            paper_id = paper_data.get('paper_id', 'unknown')
            full_text = paper_data.get('full_text', '')
            
            if not full_text:
                print(f"No text to index for paper {paper_id}")
                return False
            
            # Split into chunks
            chunks = self.split_text_into_chunks(full_text, chunk_size=1000, overlap=200)
            print(f"Split paper {paper_id} into {len(chunks)} chunks")
            
            # Index each chunk
            for i, chunk_text in enumerate(chunks):
                chunk_doc = {
                    "paper_id": paper_id,
                    "title": paper_data.get('title', ''),
                    "authors": paper_data.get('authors', ''),
                    "abstract": paper_data.get('abstract', ''),
                    "full_text": full_text,
                    "chunk_text": chunk_text,
                    "chunk_embedding": self.get_embedding(chunk_text),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "publication_date": paper_data.get('publication_date', ''),
                    "url": paper_data.get('url', ''),
                    "file_path": paper_data.get('file_path', '')
                }
                
                # Create unique document ID
                doc_id = f"{paper_id}_chunk_{i}"
                
                # Index the document
                self.client.index(
                    index=self.index_name,
                    id=doc_id,
                    body=chunk_doc
                )
            
            print(f"Successfully indexed paper {paper_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error indexing paper {paper_id}: {e}")
            return False
    
    def index_papers_from_directory(self, papers_dir: str = "papers/processed") -> Dict[str, Any]:
        """Index all processed papers from directory."""
        papers_path = Path(papers_dir)
        
        if not papers_path.exists():
            print(f"Papers directory {papers_dir} does not exist")
            return {"success": False, "error": "Directory not found"}
        
        # Find all JSON files
        json_files = list(papers_path.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {papers_dir}")
            return {"success": False, "error": "No papers to index"}
        
        print(f"Found {len(json_files)} papers to index")
        
        indexed_count = 0
        failed_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
                if self.index_paper(paper_data):
                    indexed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                failed_count += 1
        
        result = {
            "success": True,
            "total_papers": len(json_files),
            "indexed_papers": indexed_count,
            "failed_papers": failed_count
        }
        
        print(f"Indexing complete: {indexed_count} successful, {failed_count} failed")
        return result
    
    def wait_for_opensearch(self, max_retries: int = 30, delay: int = 2):
        """Wait for OpenSearch to be ready."""
        print("Waiting for OpenSearch to be ready...")
        
        for attempt in range(max_retries):
            try:
                response = self.client.cluster.health()
                if response['status'] in ['yellow', 'green']:
                    print("OpenSearch is ready!")
                    return True
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries}: OpenSearch not ready yet ({e})")
            
            time.sleep(delay)
        
        print("OpenSearch failed to become ready")
        return False

def main():
    """Main function to run the indexer."""
    print("Starting PDF Indexer...")
    
    # Wait for OpenSearch to be ready
    indexer = PDFIndexer()
    if not indexer.wait_for_opensearch():
        print("Failed to connect to OpenSearch. Make sure Docker services are running.")
        return
    
    # Index papers
    result = indexer.index_papers_from_directory()
    
    if result["success"]:
        print(f"✅ Successfully indexed {result['indexed_papers']} papers")
        if result["failed_papers"] > 0:
            print(f"⚠️  {result['failed_papers']} papers failed to index")
    else:
        print(f"❌ Indexing failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
