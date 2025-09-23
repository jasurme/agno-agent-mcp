#!/bin/bash

echo "=== Starting MCP Demo Services ==="

if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

mkdir -p data/opensearch
mkdir -p logs

echo "Starting OpenSearch and LiteLLM services..."

docker-compose up -d

echo "Waiting for services to be ready..."

echo "Waiting for OpenSearch..."
until curl -s http://localhost:9200 > /dev/null; do
    echo "Waiting for OpenSearch to start..."
    sleep 5
done

echo "OpenSearch is ready!"

echo "Waiting for LiteLLM proxy..."
until curl -s http://localhost:4000/health > /dev/null; do
    echo "Waiting for LiteLLM proxy to start..."
    sleep 5
done

echo "LiteLLM proxy is ready!"

echo ""
echo "=== Services Status ==="
echo "OpenSearch: http://localhost:9200"
echo "OpenSearch Dashboards: http://localhost:5601"
echo "LiteLLM Proxy: http://localhost:4000"
echo ""
echo "To stop services, run: docker-compose down"
echo "To view logs, run: docker-compose logs -f"
