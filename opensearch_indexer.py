#!/usr/bin/env python3
"""
OpenSearch Indexer with BGE Embeddings
Indexes processed papers in OpenSearch using BGE embeddings for semantic search.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer
import requests

class OpenSearchIndexer:
    def __init__(self, opensearch_host: str = "localhost", opensearch_port: int = 9200):
        """Initialize OpenSearch client and BGE model."""
        self.opensearch_host = opensearch_host
        self.opensearch_port = opensearch_port
        
        self.client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection,
            timeout=60
        )
        

        print("Loading BGE embedding model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        print("BGE model loaded successfully!")
        
        self.index_name = "arxiv_papers"
    
    def wait_for_opensearch(self, max_retries: int = 30):
        """Wait for OpenSearch to be ready."""
        print("Waiting for OpenSearch to be ready...")
        for i in range(max_retries):
            try:
                response = self.client.info()
                print("OpenSearch is ready!")
                return True
            except Exception as e:
                print(f"Attempt {i+1}/{max_retries}: OpenSearch not ready yet...")
                time.sleep(2)
        raise Exception("OpenSearch failed to start within the timeout period")
    
    def create_index(self):
        """Create the OpenSearch index with proper mapping."""
        print(f"Creating index: {self.index_name}")
        
        mapping = {
            "mappings": {
                "properties": {
                    "paper_id": {"type": "keyword"},
                    "full_text": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "chunks": {
                        "type": "nested",
                        "properties": {
                            "text": {"type": "text", "analyzer": "standard"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": 768,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "nmslib",
                                    "parameters": {
                                        "ef_construction": 128,
                                        "m": 24
                                    }
                                }
                            }
                        }
                    },
                    "word_count": {"type": "integer"},
                    "chunk_count": {"type": "integer"}
                }
            },
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
        }
        
        try:
            if self.client.indices.exists(index=self.index_name):
                print(f"Deleting existing index: {self.index_name}")
                self.client.indices.delete(index=self.index_name)
            
            self.client.indices.create(index=self.index_name, body=mapping)
            print(f"Index '{self.index_name}' created successfully!")
            
        except Exception as e:
            print(f"Error creating index: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def index_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Index a single paper in OpenSearch."""
        try:
            paper_id = paper_data['paper_id']
            print(f"Indexing paper: {paper_id}")
            
            chunks = paper_data.get('chunks', [])
            if chunks:
                chunk_embeddings = self.generate_embeddings(chunks)
                
                chunk_docs = []
                for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    chunk_docs.append({
                        "text": chunk_text,
                        "embedding": embedding
                    })
            else:
                chunk_docs = []
            
            doc = {
                "paper_id": paper_id,
                "full_text": paper_data.get('full_text', ''),
                "chunks": chunk_docs,
                "word_count": paper_data.get('word_count', 0),
                "chunk_count": paper_data.get('chunk_count', 0)
            }
            
            response = self.client.index(
                index=self.index_name,
                id=paper_id,
                body=doc
            )
            
            print(f"Successfully indexed: {paper_id}")
            return True
            
        except Exception as e:
            print(f"Error indexing paper {paper_id}: {e}")
            return False
    
    def index_all_papers(self, processed_dir: str):
        """Index all processed papers."""
        processed_path = Path(processed_dir)
        
        if not processed_path.exists():
            print(f"Processed directory not found: {processed_dir}")
            return
        
        json_files = list(processed_path.glob("*_processed.json"))
        
        if not json_files:
            print("No processed papers found!")
            return
        
        print(f"Found {len(json_files)} processed papers to index")
        
        successful_indexes = 0
        failed_indexes = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
                if self.index_paper(paper_data):
                    successful_indexes += 1
                else:
                    failed_indexes += 1
                    
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                failed_indexes += 1
        
        print(f"\n=== Indexing Summary ===")
        print(f"Successfully indexed: {successful_indexes}")
        print(f"Failed to index: {failed_indexes}")
        print(f"Total papers: {len(json_files)}")
    
    def test_search(self):
        """Test basic search functionality."""
        print("\n=== Testing Search Functionality ===")
        
        print("1. Testing basic text search...")
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "match": {
                            "full_text": "space"
                        }
                    },
                    "size": 3
                }
            )
            print(f"Found {response['hits']['total']['value']} documents matching 'space'")
            for hit in response['hits']['hits']:
                print(f"  - {hit['_source']['paper_id']}")
        except Exception as e:
            print(f"Basic search failed: {e}")
        
        print("\n2. Testing vector search...")
        try:
            test_query = "space exploration missions"
            query_embedding = self.embedding_model.encode([test_query])[0].tolist()
            
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
                                        "k": 3
                                    }
                                }
                            }
                        }
                    },
                    "size": 3
                }
            )
            print(f"Found {response['hits']['total']['value']} documents via vector search")
            for hit in response['hits']['hits']:
                print(f"  - {hit['_source']['paper_id']}")
        except Exception as e:
            print(f"Vector search failed: {e}")

def main():
    """Main function to set up and run the indexer."""
    print("=== OpenSearch Indexer with BGE Embeddings ===")
    
    indexer = OpenSearchIndexer()
    
    indexer.wait_for_opensearch()
    
    indexer.create_index()
    
    indexer.index_all_papers("papers/processed")
    
    indexer.test_search()
    
    print("\n=== Indexing Complete ===")
    print(f"Index: {indexer.index_name}")
    print(f"OpenSearch: http://{indexer.opensearch_host}:{indexer.opensearch_port}")
    print("You can now search the indexed papers!")

if __name__ == "__main__":
    main()


