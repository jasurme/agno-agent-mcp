
import asyncio
import json
import subprocess
import sys
import requests
from typing import Dict, Any, List

LITELLM_PROXY_URL = "http://localhost:4000"

class AgnoAgent:
    def __init__(self):
        """Initialize the Agno agent with FastMCP tools."""
        self.mcp_server_process = None
        self.request_id = 0
    
    async def start_mcp_server(self):
        """Start the FastMCP server as a subprocess."""
        try:
          
            self.mcp_server_process = subprocess.Popen(
                [sys.executable, "fastmcp_server.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for the server to start
            await asyncio.sleep(1)
            
            # Initialize the MCP server
            await self._initialize_mcp_server()
            
            print("FastMCP server started and initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to start FastMCP server: {e}")
            return False
    
    def stop_mcp_server(self):
        """Stop the FastMCP server process."""
        if self.mcp_server_process:
            self.mcp_server_process.terminate()
            self.mcp_server_process.wait()
    
    def _get_next_request_id(self) -> int:
        """Get next request ID for MCP protocol."""
        self.request_id += 1
        return self.request_id
    
    async def _initialize_mcp_server(self):
        """Initialize the MCP server."""
        if not self.mcp_server_process:
            raise Exception("MCP server not started")
        
        # Initialize the MCP server
        init_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "agno-agent",
                    "version": "1.0.0"
                }
            }
        }
        
        try:
            # Send initialization request
            request_json = json.dumps(init_request) + "\n"
            self.mcp_server_process.stdin.write(request_json)
            self.mcp_server_process.stdin.flush()
            
            # Read initialization response
            response_line = self.mcp_server_process.stdout.readline()
            if not response_line:
                raise Exception("No initialization response from MCP server")
            
            response = json.loads(response_line.strip())
            
            if "error" in response:
                raise Exception(f"MCP server initialization error: {response['error']}")
            
            # Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            
            notification_json = json.dumps(initialized_notification) + "\n"
            self.mcp_server_process.stdin.write(notification_json)
            self.mcp_server_process.stdin.flush()
            
            return True
            
        except Exception as e:
            raise Exception(f"Error initializing MCP server: {str(e)}")
    
    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the FastMCP server using MCP protocol."""
        if not self.mcp_server_process:
            raise Exception("MCP server not started")
        
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            # Send request to MCP server
            request_json = json.dumps(request) + "\n"
            self.mcp_server_process.stdin.write(request_json)
            self.mcp_server_process.stdin.flush()
            
            # Read response from MCP server
            response_line = self.mcp_server_process.stdout.readline()
            if not response_line:
                raise Exception("No response from MCP server")
            
            response = json.loads(response_line.strip())
            
            if "error" in response:
                raise Exception(f"MCP server error: {response['error']}")
            
            # FastMCP returns the result directly as a string
            result = response.get("result", {})
            
            if isinstance(result, dict) and "content" in result:
                return result["content"]
            elif isinstance(result, dict) and "text" in result:
                return result["text"]
            elif isinstance(result, str):
                # Try to parse as JSON if it looks like JSON
                if result.strip().startswith('[') or result.strip().startswith('{'):
                    try:
                        return json.loads(result)
                    except json.JSONDecodeError:
                        return result
                return result
            elif isinstance(result, list) and len(result) > 0:
                # FastMCP returns a list with one item containing the actual result
                return result[0]
            else:
                return str(result)
            
        except Exception as e:
            raise Exception(f"Error calling MCP tool {tool_name}: {str(e)}")
    
    async def bm25_search(self, query: str, size: int = 3) -> List[Dict]:
        """Perform BM25 search via FastMCP server."""
        result = await self._call_mcp_tool("bm25_search", {"query": query, "size": size})
        try:
            if isinstance(result, str):
                parsed_result = json.loads(result)
                if isinstance(parsed_result, list):
                    return parsed_result
                elif isinstance(parsed_result, dict) and "error" in parsed_result:
                    print(f"Error in BM25 search: {parsed_result['error']}")
                    return []
                else:
                    return []
            elif isinstance(result, list):
                return result
            else:
                return []
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing BM25 search result: {e}")
            return []
    
    async def vector_search(self, query: str, size: int = 3) -> List[Dict]:
        """Perform vector search via FastMCP server."""
        result = await self._call_mcp_tool("vector_search", {"query": query, "size": size})
        try:
            if isinstance(result, str):
                parsed_result = json.loads(result)
                if isinstance(parsed_result, list):
                    return parsed_result
                elif isinstance(parsed_result, dict) and "error" in parsed_result:
                    print(f"Error in vector search: {parsed_result['error']}")
                    return []
                else:
                    return []
            elif isinstance(result, list):
                return result
            else:
                return []
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing vector search result: {e}")
            return []
    
    async def hybrid_search(self, query: str, size: int = 3) -> List[Dict]:
        """Perform hybrid search via FastMCP server."""
        result = await self._call_mcp_tool("hybrid_search", {"query": query, "size": size})
        try:
            if isinstance(result, str):
                parsed_result = json.loads(result)
                if isinstance(parsed_result, list):
                    return parsed_result
                elif isinstance(parsed_result, dict) and "error" in parsed_result:
                    print(f"Error in hybrid search: {parsed_result['error']}")
                    return []
                else:
                    return []
            elif isinstance(result, list):
                return result
            else:
                return []
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing hybrid search result: {e}")
            return []
    
    async def process_user_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through three different search types:
        1. BM25 search
        2. Dense vector search  
        3. Hybrid search
        """
        print(f"\nProcessing query: '{query}'")
        print("=" * 60)
        
        try:
            print("\nBM25 Search Results:")
            print("-" * 30)
            bm25_results = await self.bm25_search(query, size=3)
            bm25_chunks = self._extract_chunks(bm25_results, "BM25")
            
            print("\nDense Vector Search Results:")
            print("-" * 30)
            dense_results = await self.vector_search(query, size=3)
            dense_chunks = self._extract_chunks(dense_results, "Dense Vector")
            
            print("\nHybrid Search Results:")
            print("-" * 30)
            hybrid_results = await self.hybrid_search(query, size=3)
            hybrid_chunks = self._extract_chunks(hybrid_results, "Hybrid")
            
           
            if hybrid_results:
                context = self._build_context_from_results(hybrid_results, query)
                llm_response = self._generate_response(context, query)
            else:
                llm_response = "No relevant documents found for LLM processing."
            
            return {
                "status": "success",
                "query": query,
                "bm25_chunks": bm25_chunks,
                "dense_chunks": dense_chunks,
                "hybrid_chunks": hybrid_chunks,
                "llm_response": llm_response
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}",
                "query": query
            }
    
    def _extract_chunks(self, search_results: List[Dict], search_type: str) -> List[str]:
        """Extract text chunks from search results."""
        if not search_results:
            print(f"No results found for {search_type}")
            return []
        
        chunks = []
        for i, hit in enumerate(search_results, 1):
            try:
                # Handle different result formats
                if '_source' in hit:
                    source = hit['_source']
                    full_text = source['full_text']
                    score = hit.get('_score', 0.0)
                elif 'source' in hit:
                    source = hit['source']
                    full_text = source['full_text']
                    score = hit.get('score', 0.0)
                elif 'full_text' in hit:
                    # Direct format
                    full_text = hit['full_text']
                    score = hit.get('score', 0.0)
                else:
                    print(f"Unexpected result format: {hit}")
                    continue
                
                chunks.append(full_text)  
                display_text = full_text[:300] + "..." if len(full_text) > 300 else full_text
                print(f"  {i}. [{score:.2f}] {display_text}")  
                
            except Exception as e:
                print(f"Error processing hit {i}: {e}")
                print(f"Hit data: {hit}")
                continue
        
        return chunks
    
    def _build_context_from_results(self, search_results: List[Dict], query: str) -> str:
        """Build context from search results for LLM."""
        context = f"Query: {query}\n\n"
        context += "Relevant Documents:\n\n"
        
        for i, hit in enumerate(search_results, 1):
            try:
                # Handle different result formats
                if '_source' in hit:
                    source = hit['_source']
                    paper_id = source.get('paper_id', f'Document {i}')
                    full_text = source['full_text']
                    score = hit.get('_score', 0.0)
                elif 'source' in hit:
                    source = hit['source']
                    paper_id = source.get('paper_id', f'Document {i}')
                    full_text = source['full_text']
                    score = hit.get('score', 0.0)
                elif 'full_text' in hit:
                    # Direct format
                    paper_id = hit.get('paper_id', f'Document {i}')
                    full_text = hit['full_text']
                    score = hit.get('score', 0.0)
                else:
                    continue
                
                context += f"Document {i}: {paper_id}\n"
                context += f"Text: {full_text}\n"
                context += f"Relevance Score: {score:.2f}\n\n"
                
            except Exception as e:
                print(f"Error building context for hit {i}: {e}")
                continue
        
        return context
    
    def _build_context(self, documents: List[Dict], query: str) -> str:
        """Build context from search results for LLM."""
        
        context ="Relevant Documents:\n\n"
        for doc in documents:
            context += f"Text: {doc['text']}\n"
        return context
        
        
    
    def _generate_response(self, context: str, query: str) -> str:
        """Generate LLM response using LiteLLM proxy."""
        prompt = f"""Based on the context below, provide answer for this:  '{query}':


<context>
{context}
</context>


"""

        return self._get_llm_response(prompt)
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM via LiteLLM proxy."""
        headers = {
            "Authorization": "Bearer demo-key-123",
            "Content-Type": "application/json"
        }
        data = {
            "model": "local-llama",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200
        }
        try:
            response = requests.post(
                f"{LITELLM_PROXY_URL}/chat/completions", 
                headers=headers, 
                json=data, 
                timeout=300
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error getting LLM response: {e}")
            return "Error: Could not get response from LLM."
    
    async def run_demo(self):
        """Run the complete demo with sample queries."""
        print("Starting Agno Framework MCP Demo")
        print("=" * 60)
        
        
        if not await self.start_mcp_server():
            print("Failed to start MCP server")
            return
        sample_queries = ["when was web audio api first introduced?"]
        
        try:
            for query in sample_queries:
                result = await self.process_user_query(query)
                
                if result["status"] == "success":
                    print("\nLLM Response:")
                    print("-" * 40)
                    print(result["llm_response"])
                    print("\n" + "=" * 60)
                else:
                    print(f"Error: {result['message']}")
                
                await asyncio.sleep(1)
        
        finally:
            self.stop_mcp_server()
        

async def main():
    """Main function to run the Agno agent demo."""
    agent = AgnoAgent()
    await agent.run_demo()

if __name__ == "__main__":
    asyncio.run(main())
