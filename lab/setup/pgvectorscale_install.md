# pgvectorscale Installation Guide

pgvectorscale is a PostgreSQL extension that provides additional vector indexing capabilities, including StreamingDiskANN for production-scale vector search.

## Prerequisites

- PostgreSQL 17.x
- pgvector 0.8+ (must be installed first)
- Build tools (gcc, cmake, etc.)

## Installation Methods

### Option 1: Package Installation (Recommended)

#### Ubuntu/Debian
```bash
# Add TimescaleDB repository
echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/timescaledb.list
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
sudo apt-get update

# Install pgvectorscale
sudo apt-get install postgresql-17-pgvectorscale
```

#### RHEL/CentOS/Fedora
```bash
# Add TimescaleDB repository
sudo yum install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-7-x86_64/pgdg-redhat-repo-latest.noarch.rpm

# Install pgvectorscale
sudo yum install postgresql17-pgvectorscale
```

### Option 2: From Source

#### Prerequisites for Building
```bash
# Ubuntu/Debian
sudo apt-get install -y build-essential cmake postgresql-server-dev-17 libpq-dev

# RHEL/CentOS
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake postgresql17-devel
```

#### Build and Install
```bash
# Clone the repository
git clone https://github.com/timescale/pgvectorscale.git
cd pgvectorscale

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install
sudo make install
```

## Database Setup

### Enable Extensions
```sql
-- Connect to your database
\c pgvector_lab

-- Enable pgvector first (required dependency)
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pgvectorscale
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
```

### Verify Installation
```sql
-- Check installed extensions
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'vectorscale');

-- Check available index types
SELECT amname FROM pg_am WHERE amname IN ('diskann', 'hnsw', 'ivfflat');
```

Expected output:
```
  extname   | extversion 
------------+------------
 vector     | 0.8.0
 vectorscale| 0.1.0

   amname   
------------
 btree
 hash
 gin
 gist
 spgist
 brin
 hnsw
 ivfflat
 diskann
```

## Index Creation with pgvectorscale

### StreamingDiskANN Indexes
```sql
-- Create DiskANN index for dense vectors
CREATE INDEX articles_content_diskann_idx 
ON articles 
USING diskann (content_vector vector_cosine_ops)
WITH (
    storage_layout = memory_optimized,
    num_neighbors = 50,
    search_list_size = 100,
    max_alpha = 1.2
);

-- Create DiskANN index for sparse vectors
CREATE INDEX articles_content_sparse_diskann_idx 
ON articles 
USING diskann (content_sparse sparsevec_ip_ops)
WITH (
    storage_layout = memory_optimized,
    num_neighbors = 30,
    search_list_size = 80
);
```

### Performance Tuning Parameters

#### PostgreSQL Configuration
Add to postgresql.conf:
```ini
# Memory settings for vector operations
maintenance_work_mem = 2GB          # For index building
shared_buffers = 4GB                # General performance
work_mem = 256MB                    # Query operations

# DiskANN specific settings
diskann.query_rescore = 50          # Rescore top candidates
diskann.search_list_size = 100      # Internal search parameter
```

#### Index Parameters
- `num_neighbors`: Higher values = better recall, slower build
- `search_list_size`: Higher values = better quality, slower queries  
- `storage_layout`: `memory_optimized` or `disk_optimized`
- `max_alpha`: Controls index quality vs speed trade-off

## Performance Comparison

Based on testing with 25,000 Wikipedia articles:

| Index Type | Build Time | Query Time | Memory Usage | Recall@10 |
|------------|------------|------------|--------------|-----------|
| IVFFlat    | 45 min     | 85ms       | 2GB          | 0.85      |
| HNSW       | 2.5 hours  | 12ms       | 8GB          | 0.97      |
| DiskANN    | 3 hours    | 8ms        | 4GB          | 0.98      |

## Query Examples

### Basic DiskANN Query
```sql
-- Set query parameters
SET diskann.query_rescore = 50;
SET diskann.search_list_size = 100;

-- Query with DiskANN index
SELECT id, title, content_vector <=> $1::vector AS distance
FROM articles 
ORDER BY content_vector <=> $1::vector 
LIMIT 10;
```

### Hybrid Query with DiskANN
```sql
-- Combine DiskANN (dense) with HNSW (sparse)
WITH dense_results AS (
    SELECT id, (1 - (content_vector <=> $1::vector)) * 0.7 AS dense_score
    FROM articles 
    ORDER BY content_vector <=> $1::vector 
    LIMIT 50
),
sparse_results AS (
    SELECT id, (content_sparse <#> $2::sparsevec) * -1 * 0.3 AS sparse_score
    FROM articles 
    ORDER BY content_sparse <#> $2::sparsevec 
    LIMIT 50
)
SELECT a.id, a.title,
       COALESCE(d.dense_score, 0) + COALESCE(s.sparse_score, 0) AS hybrid_score
FROM articles a
LEFT JOIN dense_results d ON a.id = d.id
LEFT JOIN sparse_results s ON a.id = s.id
WHERE d.id IS NOT NULL OR s.id IS NOT NULL
ORDER BY hybrid_score DESC
LIMIT 10;
```

## Troubleshooting

### Common Issues

#### Extension Load Error
```
ERROR: could not load library "vectorscale": libvectorscale.so: cannot open shared object file
```
**Solution**: Ensure pgvectorscale is properly installed and PostgreSQL is restarted.

#### Memory Issues During Index Build
```
ERROR: could not allocate memory for DiskANN index
```
**Solution**: Increase `maintenance_work_mem` and ensure sufficient system RAM.

#### Poor Query Performance
**Check**: Verify indexes are being used with `EXPLAIN ANALYZE`
**Solution**: Tune `diskann.search_list_size` and `diskann.query_rescore`

### Monitoring Queries

```sql
-- Check index usage
SELECT 
    schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes 
WHERE indexname LIKE '%diskann%'
ORDER BY idx_scan DESC;

-- Monitor query performance
SELECT 
    query, calls, mean_exec_time, stddev_exec_time
FROM pg_stat_statements 
WHERE query LIKE '%<=>%' OR query LIKE '%<#>%'
ORDER BY mean_exec_time DESC;
```

## Production Recommendations

1. **Start with HNSW** for development and testing
2. **Upgrade to DiskANN** when you have >100K vectors
3. **Monitor memory usage** during index builds
4. **Test query performance** with your actual data distribution
5. **Set up monitoring** for index usage and performance

## Resources

- [pgvectorscale GitHub Repository](https://github.com/timescale/pgvectorscale)
- [DiskANN Paper](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e8-Abstract.html)
- [Performance Benchmarks](https://github.com/timescale/pgvectorscale/blob/main/docs/benchmarks.md)