-- One workload "tick": insert :rows row(s) with a strictly increasing n.
-- Driven in a loop by scripts/workload.sh against the HAProxy primary port,
-- so it always writes to the current leader.
--   psql ... -v rows=1 -f sql/10-workload.sql
\if :{?rows}
\else
  \set rows 1
\endif

INSERT INTO public.orders (n, source_node, note)
SELECT nextval('public.orders_n_seq'),
       COALESCE(host(inet_server_addr()), 'socket'),  -- node IP that served the write
       'tick'
FROM generate_series(1, :rows);
