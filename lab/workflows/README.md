# n8n Workflow Automation

This directory contains n8n workflows for automating and visualizing RAG operations.

## Available Workflows

### 1. Naive RAG Workflow (`naive_rag_workflow.json`)
- Basic RAG implementation with simple vector search
- Demonstrates fundamental retrieval-augmented generation
- Good starting point for understanding RAG concepts

### 2. Hybrid RAG Workflow (`hybrid_rag_workflow.json`)
- Combines dense and sparse vector search
- Interactive parameter tuning for weight adjustment
- Shows performance improvements over naive approach

### 3. Adaptive RAG Workflow (`adaptive_rag_workflow.json`)
- Intelligent query classification and routing
- Dynamic weight adjustment based on query type
- Demonstrates advanced RAG techniques

### 4. Comparison Workflow (`comparison_workflow.json`)
- Side-by-side comparison of different search methods
- Performance metrics and visualization
- Useful for demonstrating superiority of hybrid approach

## Quick Start

### Using Docker Compose (Recommended)

1. Start the complete stack:
```bash
cd lab/06_workflows
docker-compose up -d
```

2. Access n8n at: http://localhost:5678

3. Import workflows:
   - Click "Workflows" → "Import from File"
   - Select desired .json workflow file
   - Configure API endpoints in workflow nodes

### Manual n8n Setup

1. Install n8n globally:
```bash
npm install -g n8n
```

2. Start n8n:
```bash
n8n start
```

3. Import workflows as described above

## Configuration

Each workflow requires configuration of:
- PostgreSQL connection (DATABASE_URL)
- OpenAI API key (OPENAI_API_KEY)
- FastAPI endpoint (default: http://localhost:8000)

Update these in the HTTP Request nodes within each workflow.

## Workflow Descriptions

### Query Classification Logic
The adaptive workflow classifies queries into:
- **Factual**: Names, dates, specific facts (sparse-heavy: 0.3 dense, 0.7 sparse)
- **Conceptual**: Ideas, explanations, relationships (dense-heavy: 0.7 dense, 0.3 sparse)
- **Exploratory**: Open-ended, research questions (balanced: 0.5 dense, 0.5 sparse)

### Performance Metrics
All workflows track:
- Query latency (ms)
- Token usage and costs
- Similarity scores
- Result relevance

## Integration with FastAPI

Ensure the FastAPI server is running:
```bash
cd lab/05_api
python fastapi_server.py
```

The workflows communicate with these endpoints:
- `/search`: Main search endpoint
- `/compare`: Method comparison
- `/analyze`: Query analysis
- `/stats`: Statistics and metrics

## Best Practices

1. **Development**: Use n8n for rapid prototyping and visualization
2. **Production**: Extract logic to Python APIs for better performance
3. **Monitoring**: Use n8n's execution history for debugging
4. **Scaling**: Deploy multiple n8n instances behind a load balancer

## Troubleshooting

Common issues:
- **Connection refused**: Ensure FastAPI server is running
- **API key errors**: Check OPENAI_API_KEY in workflow nodes
- **Database errors**: Verify PostgreSQL connection and pgvector extension
- **Timeout errors**: Increase timeout in HTTP Request nodes

## Resources

- [n8n Documentation](https://docs.n8n.io/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Project README](../../README.md)