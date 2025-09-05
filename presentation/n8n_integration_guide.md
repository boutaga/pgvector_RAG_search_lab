# n8n Integration Guide for RAG Demo
## Complete Setup and Workflow Templates

### **Prerequisites**
- Docker installed and running
- Your existing lab APIs running (FastAPI from lab/05_api/)
- Environment variables configured
- n8n community edition (free)

---

## **1. n8n Setup**

### **Docker Installation (Recommended)**
```bash
# Create n8n directory structure
mkdir -p ~/n8n/data
cd ~/n8n

# Create docker-compose.yml for n8n + PostgreSQL
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  n8n:
    image: n8nio/n8n:latest
    container_name: n8n_rag_demo
    restart: unless-stopped
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=false
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - WEBHOOK_URL=http://localhost:5678/
      - GENERIC_TIMEZONE=UTC
      # Enable expressions and code node
      - N8N_SECURE_COOKIE=false
      - N8N_DISABLE_UI=false
    volumes:
      - ./data:/home/node/.n8n
      - /var/run/docker.sock:/var/run/docker.sock
    networks:
      - n8n-network

  # Optional: Local PostgreSQL for n8n (if not using existing)
  postgres-n8n:
    image: ankane/pgvector:latest
    container_name: postgres_n8n
    restart: unless-stopped
    environment:
      POSTGRES_DB: n8n_demo
      POSTGRES_USER: n8n_user
      POSTGRES_PASSWORD: n8n_password
    ports:
      - "5433:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - n8n-network

networks:
  n8n-network:
    driver: bridge

volumes:
  postgres_data:
EOF

# Start n8n
docker-compose up -d

# Access n8n at http://localhost:5678
echo "n8n is starting... Access it at http://localhost:5678"
```

### **Environment Configuration**
Create `.env` file in your lab directory:
```bash
# Database connection  
DATABASE_URL=postgresql://postgres:password@localhost:5432/wikipedia
DATABASE_URL_N8N=postgresql://n8n_user:n8n_password@localhost:5433/n8n_demo

# OpenAI API
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL_EMBED=text-embedding-3-small
OPENAI_MODEL_CHAT=gpt-4o-mini

# Your lab API endpoints
LAB_API_BASE=http://host.docker.internal:8000
# Note: Use host.docker.internal from inside n8n container to reach host APIs

# n8n configuration
N8N_ENCRYPTION_KEY=your-encryption-key-here
```

---

## **2. FastAPI Endpoints for n8n**

Create enhanced API endpoints optimized for n8n integration:

```python
# lab/05_api/n8n_endpoints.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import time
import asyncio

app = FastAPI(title="RAG API for n8n Integration")

class SearchRequest(BaseModel):
    query: str
    method: str = "hybrid"  # naive, hybrid, adaptive
    top_k: int = 5
    alpha: Optional[float] = 0.5  # for hybrid method
    include_metadata: bool = True
    include_timing: bool = True

class SearchResponse(BaseModel):
    answer: str
    method_used: str
    query: str
    contexts: List[Dict]
    metadata: Dict[str, Any]
    timing: Dict[str, float]
    success: bool = True

@app.post("/search", response_model=SearchResponse)
async def unified_search(request: SearchRequest):
    """Unified search endpoint for all RAG methods"""
    start_time = time.time()
    
    try:
        # Route to appropriate method
        if request.method == "naive":
            result = await search_naive(request)
        elif request.method == "hybrid": 
            result = await search_hybrid(request)
        elif request.method == "adaptive":
            result = await search_adaptive(request)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        # Add timing information
        total_time = time.time() - start_time
        result["timing"] = {
            "total_ms": round(total_time * 1000, 2),
            "method": request.method
        }
        
        result["success"] = True
        return SearchResponse(**result)
        
    except Exception as e:
        return SearchResponse(
            answer=f"Error: {str(e)}",
            method_used=request.method,
            query=request.query,
            contexts=[],
            metadata={"error": str(e)},
            timing={"total_ms": round((time.time() - start_time) * 1000, 2)},
            success=False
        )

@app.post("/compare")
async def compare_methods(request: SearchRequest):
    """Compare all three methods side by side"""
    methods = ["naive", "hybrid", "adaptive"]
    results = {}
    
    for method in methods:
        method_request = request.copy()
        method_request.method = method
        results[method] = await unified_search(method_request)
    
    return {
        "query": request.query,
        "comparison": results,
        "winner": max(results.keys(), 
                     key=lambda k: len(results[k].get("contexts", [])))
    }

@app.get("/health")
async def health_check():
    """Health check for n8n monitoring"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "available_methods": ["naive", "hybrid", "adaptive", "compare"]
    }

# Method implementations (using your existing lab code)
async def search_naive(request: SearchRequest):
    # Import and use your existing naive_search code
    from lab.04_search.simple_search import SimpleSearch
    search = SimpleSearch()
    return search.search_dense(request.query, top_k=request.top_k)

async def search_hybrid(request: SearchRequest):
    # Import and use your existing hybrid_search code  
    from lab.04_search.hybrid_search import HybridSearch
    search = HybridSearch()
    return search.search_weighted(
        request.query, 
        dense_weight=request.alpha,
        sparse_weight=1.0-request.alpha,
        top_k=request.top_k
    )

async def search_adaptive(request: SearchRequest):
    # Import and use your existing adaptive_search code
    from lab.04_search.adaptive_search import AdaptiveSearch  
    search = AdaptiveSearch()
    return search.search_adaptive(request.query, top_k=request.top_k)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## **3. n8n Workflow Templates**

### **Workflow 1: Naive RAG (Starting Point)**

**Nodes:**
1. **Manual Trigger** → Start workflow manually
2. **Set Variables** → Configure query and parameters
3. **HTTP Request** → Call your API
4. **JSON Parser** → Extract response fields
5. **Display Results** → Show formatted output

**JSON Export for Import:**
```json
{
  "name": "Naive RAG Demo",
  "nodes": [
    {
      "id": "manual-trigger",
      "name": "Demo Trigger",
      "type": "n8n-nodes-base.manualTrigger",
      "position": [250, 300],
      "parameters": {}
    },
    {
      "id": "set-query",
      "name": "Set Query & Config", 
      "type": "n8n-nodes-base.set",
      "position": [450, 300],
      "parameters": {
        "values": {
          "string": [
            {
              "name": "query",
              "value": "What is WAL in PostgreSQL and why is it important?"
            },
            {
              "name": "method", 
              "value": "naive"
            },
            {
              "name": "top_k",
              "value": "5"
            }
          ]
        }
      }
    },
    {
      "id": "api-call",
      "name": "RAG Search API",
      "type": "n8n-nodes-base.httpRequest",
      "position": [650, 300],
      "parameters": {
        "method": "POST",
        "url": "http://host.docker.internal:8000/search",
        "options": {
          "headers": {
            "Content-Type": "application/json"
          }
        },
        "bodyParameters": {
          "query": "={{ $json.query }}",
          "method": "={{ $json.method }}",
          "top_k": "={{ parseInt($json.top_k) }}",
          "include_metadata": true,
          "include_timing": true
        }
      }
    },
    {
      "id": "format-results",
      "name": "Format Output",
      "type": "n8n-nodes-base.function",
      "position": [850, 300], 
      "parameters": {
        "functionCode": "// Extract and format the API response\nconst response = items[0].json;\n\n// Create formatted output\nconst output = {\n  query: response.query,\n  method: response.method_used,\n  answer: response.answer,\n  timing_ms: response.timing?.total_ms || 0,\n  context_count: response.contexts?.length || 0,\n  top_contexts: (response.contexts || []).slice(0, 3).map(ctx => ({\n    title: ctx.title,\n    similarity: ctx.sim_dense || ctx.similarity || 0,\n    snippet: (ctx.content || '').substring(0, 200) + '...'\n  })),\n  success: response.success\n};\n\nreturn [{ json: output }];"
      }
    }
  ],
  "connections": {
    "manual-trigger": {"main": [[{"node": "set-query", "type": "main", "index": 0}]]},
    "set-query": {"main": [[{"node": "api-call", "type": "main", "index": 0}]]},
    "api-call": {"main": [[{"node": "format-results", "type": "main", "index": 0}]]}
  }
}
```

### **Workflow 2: Hybrid RAG with Parameter Control**

**Enhanced Features:**
- Alpha slider for dense/sparse weighting
- Side-by-side comparison
- Performance timing display

**Key Nodes:**
```json
{
  "name": "Hybrid RAG with Controls",
  "nodes": [
    {
      "id": "manual-trigger-advanced",
      "name": "Advanced Demo Trigger",
      "type": "n8n-nodes-base.manualTrigger",
      "position": [200, 300],
      "parameters": {}
    },
    {
      "id": "set-parameters",
      "name": "Set Parameters",
      "type": "n8n-nodes-base.set",
      "position": [400, 300],
      "parameters": {
        "values": {
          "string": [
            {
              "name": "query",
              "value": "How does MVCC reduce locking in PostgreSQL?"
            }
          ],
          "number": [
            {
              "name": "alpha",
              "value": 0.5
            },
            {
              "name": "top_k", 
              "value": 5
            }
          ]
        }
      }
    },
    {
      "id": "parallel-methods",
      "name": "Test Multiple Methods",
      "type": "n8n-nodes-base.splitInBatches",
      "position": [600, 300],
      "parameters": {
        "batchSize": 1,
        "options": {}
      }
    },
    {
      "id": "method-switch",
      "name": "Method Router",
      "type": "n8n-nodes-base.switch",
      "position": [800, 300],
      "parameters": {
        "conditions": {
          "options": {
            "mode": "list"
          },
          "values": [
            {
              "operation": "equal",
              "value1": "={{ $json.method }}",
              "value2": "naive"
            },
            {
              "operation": "equal", 
              "value1": "={{ $json.method }}",
              "value2": "hybrid"
            },
            {
              "operation": "equal",
              "value1": "={{ $json.method }}",
              "value2": "adaptive"
            }
          ]
        },
        "fallbackOutput": "extra"
      }
    }
  ]
}
```

### **Workflow 3: Full Comparison Dashboard**

**Features:**
- Automatic method comparison
- Performance benchmarking
- Result quality scoring
- Visual output formatting

---

## **4. Live Demo Script**

### **Demo Setup (Pre-conference)**
1. **Start all services:**
```bash
# Terminal 1: Start your lab API
cd lab/05_api
source ../../venv/bin/activate
python n8n_endpoints.py

# Terminal 2: Start n8n
cd ~/n8n
docker-compose up -d

# Terminal 3: Monitor logs
docker logs -f n8n_rag_demo
```

2. **Test endpoints:**
```bash
# Test API availability
curl http://localhost:8000/health

# Test search functionality
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is WAL in PostgreSQL?", "method": "naive"}'
```

3. **Prepare n8n workspace:**
- Import workflow templates
- Set up credentials (OpenAI API key)
- Test basic connectivity

### **Live Demo Flow (12 minutes)**

#### **Minute 1-2: Environment Setup**
**Audience Script:**
"Let me show you how quickly we can prototype RAG systems with n8n. I've got my PostgreSQL database running with pgvector, and my Python APIs ready. Let's start from scratch."

**Actions:**
- Open n8n interface (http://localhost:5678)
- Show empty workspace
- Briefly explain n8n UI (nodes, connections, execution)

#### **Minute 3-5: Build Naive RAG**
**Audience Script:**
"First, let's build the simplest possible RAG system - what most tutorials show you. Query → Embed → Search → LLM."

**Actions:**
1. Drag **Manual Trigger** node
2. Add **Set** node for query parameters:
   - `query`: "What is WAL in PostgreSQL?" 
   - `method`: "naive"
   - `top_k`: 5
3. Add **HTTP Request** node:
   - URL: `http://host.docker.internal:8000/search`
   - Method: POST
   - Body: Use expressions to pass parameters
4. Connect nodes and execute
5. Show JSON output in execution view

**Key Points:**
- "This works, but see how it missed some acronym-specific content?"
- "Response time: ~200ms, decent but not great relevance"

#### **Minute 6-9: Upgrade to Hybrid**
**Audience Script:**
"Now let's see the hybrid approach in action. Same query, but we'll use both dense and sparse embeddings."

**Actions:**
1. Duplicate the workflow
2. Change method parameter to "hybrid"
3. Add alpha parameter (0.5 - balanced)
4. Execute and compare results
5. **Interactive tuning:** Change alpha to 0.3 (more sparse), then 0.7 (more dense)
6. Show result differences in real-time

**Key Points:**
- "Alpha = 0.3 (sparse-heavy): Better exact term matching" 
- "Alpha = 0.7 (dense-heavy): Better conceptual understanding"
- "Notice how contexts change with different weightings"

#### **Minute 10-11: Adaptive Intelligence**  
**Audience Script:**
"But we shouldn't manually tune for every query. Let's add intelligence - the system decides the best approach based on query characteristics."

**Actions:**
1. Create new workflow branch
2. Set method to "adaptive" 
3. Execute with different query types:
   - Factual: "What is WAL in PostgreSQL?"
   - Conceptual: "How does MVCC work?"
   - Procedural: "How to configure WAL archiving?"
4. Show how method adapts automatically

**Key Points:**
- "System automatically detects query type"
- "Routes to optimal search strategy" 
- "15-20% improvement in relevance without manual tuning"

#### **Minute 12: Production Reality Check**
**Audience Script:**
"This looks great for demos, but what about production? Here's where n8n shines AND where you need to be careful."

**Actions:**
1. Show `/compare` endpoint calling all methods simultaneously
2. Display performance metrics (timing, cost)
3. Point out API cost implications
4. Quick Docker show for deployment

**Key Points:**
- "n8n great for: prototyping, business logic workflows, stakeholder demos"
- "Extract to Python APIs for: performance, cost control, complex logic"
- "Don't fall into the LangChain abstraction trap"

---

## **5. Production Deployment Guide**

### **n8n Production Setup**
```yaml
# production-docker-compose.yml
version: '3.8'
services:
  n8n-production:
    image: n8nio/n8n:latest
    restart: unless-stopped
    ports:
      - "443:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_HOST=${N8N_DOMAIN}
      - N8N_PROTOCOL=https
      - N8N_PORT=443
      - WEBHOOK_URL=https://${N8N_DOMAIN}/
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_DATABASE=n8n_prod
      - DB_POSTGRESDB_USER=${POSTGRES_USER}
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD}
      - N8N_ENCRYPTION_KEY=${N8N_ENCRYPTION_KEY}
    volumes:
      - n8n_storage:/home/node/.n8n
      - /etc/ssl/certs:/etc/ssl/certs:ro
    networks:
      - production-network

  postgres-prod:
    image: ankane/pgvector:latest
    restart: unless-stopped
    environment:
      POSTGRES_DB: n8n_prod
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
    networks:
      - production-network

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl:ro
    depends_on:
      - n8n-production
    networks:
      - production-network

networks:
  production-network:
    driver: bridge

volumes:
  n8n_storage:
  postgres_prod_data:
```

### **Security Considerations**
- Enable authentication (basic auth or OAuth)
- Use HTTPS in production
- Secure API keys with n8n credentials
- Set up proper firewall rules
- Regular backups of workflow data

### **Scaling Strategy**
1. **Development**: n8n for all workflow logic
2. **Staging**: n8n orchestration + Python APIs for heavy lifting  
3. **Production**: n8n for business workflows only, Python microservices for RAG operations
4. **Enterprise**: Multiple n8n instances, load balancing, separate database clusters

---

## **6. Troubleshooting Guide**

### **Common Issues**

**"Connection refused" errors:**
- Check `host.docker.internal` resolution
- Verify API is running on correct port
- Ensure Docker networking is configured properly

**Slow response times:**
- Check PostgreSQL connection pooling
- Monitor OpenAI API rate limits
- Verify index usage in queries

**Memory issues in n8n:**
- Increase Docker memory allocation
- Optimize large data processing nodes
- Use streaming where possible

**API key errors:**
- Verify OpenAI key is properly set in credentials
- Check key permissions and quotas
- Monitor usage billing

### **Performance Optimization**
- Cache frequent embeddings
- Use connection pooling
- Batch API requests where possible
- Monitor and tune PostgreSQL queries
- Set up proper indexes

This guide provides everything you need for a smooth n8n demo and production deployment path!