# Live Demo Script: "What the RAG?" n8n Workshop
## Detailed Step-by-Step Guide with Timing

**Total Demo Time:** 15 minutes  
**Backup Plan:** Pre-recorded screencast available if technical issues  
**Recovery Strategy:** Each section has checkpoint saves

---

## **Pre-Demo Setup Checklist (30 minutes before talk)**

### **Environment Status Check**
```bash
# 1. Verify PostgreSQL + pgvector
psql -d wikipedia -c "SELECT COUNT(*) FROM articles WHERE dense IS NOT NULL;"
psql -d wikipedia -c "SELECT COUNT(*) FROM articles WHERE sparse IS NOT NULL;"

# 2. Check API endpoints  
curl http://localhost:8000/health
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "method": "naive"}'

# 3. Verify n8n access
curl http://localhost:5678/rest/login

# 4. Test queries ready
export TEST_QUERIES=("What is WAL in PostgreSQL?" "How does MVCC work?" "Steps to configure replication")
```

### **Screen Setup**
- **Primary Display:** n8n interface (localhost:5678) 
- **Secondary Display:** API responses / terminal logs
- **Browser Tabs Ready:**
  - n8n workflows (clean workspace)
  - Backup workflows (in case of issues)
  - API documentation (if needed)

### **Demo Props**
- Clicker/presenter remote
- Backup laptop with identical setup
- Mobile hotspot (in case of network issues)
- Pre-recorded screencast (absolute fallback)

---

## **Demo Script with Exact Timing**

### **📱 MINUTE 1-2: Opening & Environment (2 min)**

#### **[00:00-00:30] Introduction**
**Speaker Script:**
> "I've been promising you a live demo, and here it is. I'm going to build three different RAG systems from scratch in n8n, and show you exactly why the hybrid approach consistently outperforms everything else. If something breaks, that's real-world demo authenticity!"

**Actions:**
- Open n8n interface (should already be loaded)
- Show clean workspace with no workflows
- Quick browser check (API health endpoint)

**Audience Confirmation:**
- "Can everyone see the n8n interface clearly?"
- "I have [X] minutes for this demo, so I'll move quickly but take questions at the end"

#### **[00:30-02:00] Environment Tour**
**Speaker Script:**  
> "Behind the scenes, I have PostgreSQL with pgvector running 25,000 Wikipedia articles. Dense embeddings from OpenAI, sparse embeddings from SPLADE, and my Python APIs that we'll be calling. This is exactly what you'd build in your organization."

**Actions:**
1. Open terminal (briefly show API logs)
2. Quick `curl http://localhost:8000/health` demo  
3. Back to n8n interface
4. Explain n8n basics (30 seconds max):
   - "Nodes do work, connections pass data"
   - "Visual programming for workflows"
   - "Execute button to run, view results inline"

**Key Message:** "This is production-ready infrastructure, not a toy demo."

---

### **🔍 MINUTE 3-5: Build Naive RAG (3 min)**

#### **[02:00-02:30] Create Basic Workflow**
**Speaker Script:**
> "Let's start with naive RAG - what 90% of tutorials show you. Single embedding model, simple vector search, that's it."

**Actions - Watch Screen Closely:**
1. **Drag Manual Trigger node** to workspace
   - Position: center-left  
   - Name it: "Demo Start"
   - Click "Execute Node" to test (should show green checkmark)

2. **Add Set node** for parameters
   - Drag from node library, connect to Manual Trigger
   - Name: "Query Setup"
   - Add string parameters:
     - `query`: "What is WAL in PostgreSQL and why is it important?"
     - `method`: "naive" 
     - `top_k`: "5"

**Checkpoint:** Both nodes green, connection visible

#### **[02:30-04:00] Add API Call**  
**Speaker Script:**
> "Now the magic - one HTTP request to our RAG API. This is where we would call OpenAI for embeddings, PostgreSQL for vector search, and another OpenAI call for answer generation."

**Actions:**
3. **Add HTTP Request node**
   - Connect to Set node
   - Name: "RAG Search"  
   - Configure:
     - Method: POST
     - URL: `http://host.docker.internal:8000/search`
     - Headers: `Content-Type: application/json`
     - Body (JSON):
       ```json
       {
         "query": "={{ $json.query }}",
         "method": "={{ $json.method }}",
         "top_k": "={{ parseInt($json.top_k) }}",
         "include_metadata": true
       }
       ```

**Real-Time Commentary:**
- "Notice I'm using n8n expressions - `{{ $json.query }}` pulls from previous node"
- "host.docker.internal lets n8n container reach my host APIs"

4. **Execute the workflow**
   - Click "Execute Workflow" button
   - **Wait for results** (should be 1-3 seconds)
   - Click on HTTP Request node to view JSON response

#### **[04:00-05:00] Analyze Results**
**Speaker Script:**
> "Look at these results. We got an answer, but check the contexts - mostly conceptual matches. It missed some of the detailed WAL implementation because 'WAL' is a specific acronym."

**Actions:**  
- Expand JSON response in n8n
- Point out key fields:
  - `answer`: The LLM response
  - `contexts`: Retrieved documents  
  - `timing`: Response time (~200-400ms)
  - `method_used`: "naive"

**Key Observation:** "Good semantic understanding, but weak on exact technical terms."

---

### **⚡ MINUTE 6-9: Upgrade to Hybrid RAG (4 min)**

#### **[05:00-05:30] Duplicate & Modify**
**Speaker Script:**
> "Now let's see what happens when we add sparse embeddings. Same query, same infrastructure, but now we're combining semantic AND lexical search."

**Actions:**
1. **Duplicate entire workflow** (right-click → duplicate)
2. **Modify the Set node** in new workflow:
   - Change `method` from "naive" to "hybrid"
   - Add `alpha`: "0.5" (balanced weights)
3. **Rename workflow** to "Hybrid RAG Demo"

**Checkpoint:** Two workflows visible, hybrid workflow ready

#### **[05:30-07:00] Execute & Compare**
**Speaker Script:**  
> "Same exact query - 'What is WAL in PostgreSQL and why is it important?' But watch how the results change with hybrid search."

**Actions:**
4. **Execute hybrid workflow**
5. **Wait for results** (may be slightly slower - 300-600ms)  
6. **Compare side by side** (open both result JSONs)
7. **Point out differences:**
   - Different contexts retrieved
   - Look for more specific WAL-related content
   - Check similarity scores if visible

**Real-Time Analysis:**
- "Notice context #1 now - more specific WAL documentation"
- "The answer quality should be noticeably better"
- "Hybrid found both conceptual AND implementation details"

#### **[07:00-09:00] Interactive Tuning**  
**Speaker Script:**
> "Here's where it gets interesting. Let me show you the power of adjusting the dense/sparse balance in real-time. This is why you want control over your search strategy."

**Actions:**
8. **Modify alpha parameter** in Set node:
   - First: Change to `"0.3"` (sparse-heavy)
   - Execute workflow
   - Show results: "More exact term matching"
   
9. **Change alpha again:**  
   - Set to `"0.7"` (dense-heavy)
   - Execute workflow  
   - Show results: "More conceptual/semantic matches"

**Audience Engagement:**
- "Who thinks 0.3 gave better results?" [show of hands]
- "Who prefers 0.7?" [show of hands]  
- "This is exactly the kind of tuning you do in production"

**Key Message:** "Real-time parameter tuning - try doing this with a black-box LangChain setup!"

---

### **🧠 MINUTE 10-12: Add Adaptive Intelligence (3 min)**

#### **[09:00-10:00] Build Smart Routing**
**Speaker Script:**
> "But we shouldn't tune manually for every query type. Let's add intelligence - the system automatically picks the best approach based on query characteristics."

**Actions:**
1. **Create third workflow** (duplicate hybrid)
2. **Modify Set node:**
   - Change `method` to "adaptive"
   - Remove `alpha` parameter (system decides automatically)
3. **Rename:** "Adaptive RAG Demo"

#### **[10:00-12:00] Test Query Types**
**Speaker Script:**  
> "Watch how the system adapts. Different query types get routed to optimal search strategies automatically."

**Actions:**
4. **Test factual query:**  
   - Execute with "What is WAL in PostgreSQL?"
   - Show results, explain routing decision
   
5. **Change to conceptual query:**
   - Modify query to: "How does MVCC reduce locking?"
   - Execute, show different approach taken
   
6. **Try procedural query:**
   - Query: "Steps to configure WAL archiving"
   - Execute, demonstrate third routing pattern

**Real-Time Commentary for Each:**
- "Factual query → sparse-heavy weights (0.3/0.7)"
- "Conceptual query → dense-heavy weights (0.7/0.3)"  
- "Procedural query → balanced approach (0.5/0.5)"

**Key Message:** "15-20% improvement in relevance with zero manual tuning!"

---

### **🚀 MINUTE 13-15: Production Reality & Wrap-up (3 min)**

#### **[12:00-13:30] Production Considerations**
**Speaker Script:**
> "This demo looks great, but let's talk production reality. n8n is fantastic for prototyping and business workflows, but you need to be strategic about what lives where."

**Actions:**
1. **Show comparison endpoint:** 
   - Quick demo of `/compare` API call
   - Display all three methods side-by-side
   - Point out performance/cost differences
   
2. **Open Docker Compose** (briefly):
   - Show production deployment setup
   - Highlight scaling considerations

**Key Points:**
- "n8n perfect for: visual workflows, stakeholder demos, business logic"
- "Python APIs for: performance-critical operations, cost control"
- "Never put your expensive LLM calls inside complex visual workflows"

#### **[13:30-14:30] LangChain Warning**
**Speaker Script:**
> "Quick warning about LangChain - it's great for getting started, but becomes a trap in production. You lose control over API calls, costs, and debugging becomes impossible."

**Actions:**
3. **Show cost comparison** (if time):
   - "Direct API: $0.002 per query"
   - "Through LangChain abstraction: $0.008 per query"
   - "That's 4x cost increase from hidden API calls"

**Key Message:** "Abstract the periphery, own the critical path."

#### **[14:30-15:00] Demo Conclusion**
**Speaker Script:**
> "In 15 minutes, we built three production-ready RAG systems, showed why hybrid outperforms single methods, and gave you a clear path from prototype to production. The code is all available in my GitHub repo."

**Actions:**
4. **Final execution** of all three workflows simultaneously (if technically feasible)
5. **Show n8n workflow library** for export/sharing
6. **Display GitHub repo link** on screen

**Closing:** "Any questions about what we just built?"

---

## **🔧 Technical Backup Plans**

### **If n8n Fails to Load**
1. **Switch to pre-recorded screencast** (5 minutes condensed version)
2. **Live API demos** via curl commands  
3. **Jupyter notebook** backup with same workflows in Python

### **If API Endpoints Fail**
1. **Mock API responses** (pre-saved JSON files)
2. **Local SQLite database** with sample data
3. **Pure PostgreSQL** queries without AI components

### **If Network Issues**
1. **Local-only setup** (no OpenAI calls)
2. **Pre-computed embeddings** loaded from files
3. **Offline demo environment** on backup laptop

### **If Demo Runs Too Fast**
- **Extended query testing** with audience suggestions  
- **Deep dive into JSON responses** and explain each field
- **Interactive parameter tuning** with audience voting

### **If Demo Runs Too Slow**
- **Skip adaptive section**, focus on naive vs hybrid comparison
- **Pre-execute workflows**, just show results
- **Use pre-recorded sections** for time-intensive parts

---

## **🎯 Success Metrics**

### **Audience Engagement Indicators**
- **Questions during demo** (good sign - they're following)
- **Phones down** (visual attention)  
- **Note-taking** (planning to implement)

### **Technical Success Markers**  
- **All workflows execute successfully**
- **Clear performance differences shown**  
- **Real-time parameter tuning works**
- **Timing stays within 15 minutes**

### **Message Delivery Goals**
1. ✅ **Hybrid search clearly outperforms** single methods
2. ✅ **n8n useful for prototyping** but extract to APIs for production  
3. ✅ **PostgreSQL handles vector workloads** as well as specialized databases
4. ✅ **DBAs essential** for production AI systems
5. ✅ **Avoid LangChain abstraction trap** in critical paths

---

## **🗣️ Audience Interaction Points**

### **Planned Questions to Ask**
- "How many of you have tried RAG systems?" (show of hands)
- "Who thinks sparse search won that round?" (after parameter tuning)
- "Any questions about the n8n setup?" (during technical sections)
- "What query should we test next?" (if time permits)

### **Expected Questions from Audience**
- **"How does this scale?"** → Point to PostgreSQL scaling, connection pooling
- **"What about cost?"** → Show cost breakdown, API optimization strategies  
- **"Security concerns?"** → Database access controls, API key management
- **"Alternatives to n8n?"** → Airflow, Prefect, but n8n best for visual prototyping

### **Difficult Questions Preparation**  
- **"Why not just use ChatGPT?"** → Data control, cost, hallucination prevention
- **"How do you measure quality?"** → Precision/recall metrics, human evaluation
- **"What about hallucinations?"** → Source attribution, confidence scoring
- **"Performance at enterprise scale?"** → Benchmarking results, optimization strategies

---

This script gives you confidence for a smooth 15-minute demo that clearly demonstrates the hybrid RAG advantage while showing practical n8n usage!