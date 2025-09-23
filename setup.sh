#!/bin/bash

echo "🚀 MCP Demo Setup Script"
echo "========================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop first."
    echo "   Download from: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Download from: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    echo "   Download from: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Prerequisites check passed!"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo "✅ Python dependencies installed!"

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "❌ Failed to start Docker services"
    exit 1
fi

echo "✅ Docker services started!"

# Wait for services to initialize
echo "⏳ Waiting for services to initialize (this may take 2-3 minutes)..."
sleep 30

# Check if OpenSearch is ready
echo "🔍 Checking OpenSearch health..."
for i in {1..30}; do
    if curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
        echo "✅ OpenSearch is ready!"
        break
    fi
    echo "   Waiting for OpenSearch... ($i/30)"
    sleep 10
done

# Index sample PDFs
echo "📚 Indexing sample PDFs..."
python3 opensearch_indexer.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to index PDFs"
    exit 1
fi

echo "✅ PDFs indexed successfully!"

echo ""
echo "🎉 Setup complete! You can now run the demo with:"
echo "   python3 agno_agent.py"
echo ""
echo "📊 To view OpenSearch Dashboards, visit: http://localhost:5601"
echo "🔧 To stop services, run: docker-compose down"
echo ""
