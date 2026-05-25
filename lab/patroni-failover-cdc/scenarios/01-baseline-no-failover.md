# Scenario 01 — Baseline: the slot is lost on switchover (the silent break)

**Claim proven (and verified on a real run):** a normal logical slot
(`failover = false`, the default) lives only on the primary. After a Patroni
switchover the new primary does not have it, Debezium **silently recreates it at
the current position**, the connector never leaves `RUNNING`, and the rows committed
in between are **lost**. This is the quiet failure teams hit in production.

> Output below is from an actual run. Leader was `patroni1`, standby `patroni2`.

## 1. Start clean + plain connector

```bash
./scripts/reset.sh                  # drops connectors/slots, recreates schema, clears the topic
./scripts/workload.sh               # terminal B
./scripts/register-debezium.sh 01   # slot.name=dbz_slot, slot.failover=false  -> HTTP 201
```

## 2. The slot exists only on the primary

```bash
./scripts/slots.sh
```

```
===== patroni1 (LEADER) =====
 slot_name | slot_type | active | failover | synced
-----------+-----------+--------+----------+--------
 dbz_slot  | logical   | t      | f        | f
 patroni2  | physical  | t      | f        | f

===== patroni2 (replica) =====
 slot_name | slot_type | active | failover | synced
-----------+-----------+--------+----------+--------
 patroni1  | physical  | f      | f        | f
        (no dbz_slot here — failover=f, nothing mirrors it)
```

## 3. Switch over

```bash
./scripts/switchover.sh
```

```
2026-... Successfully switched over to "patroni2"
```

## 4. The slot is gone, and Debezium quietly recreates it

Right after promotion, `dbz_slot` does not exist on the new primary:

```
===== patroni2 (LEADER) =====
 slot_name | slot_type | active | failover
-----------+-----------+--------+----------
 patroni1  | physical  | t      | f
        (dbz_slot is gone — it never left the old primary)
```

But the connector status stays green the whole time:

```bash
./scripts/connect-status.sh orders-plain
```

```
{ "conn": "RUNNING", "task": "RUNNING", "trace": "" }
```

and the Connect log shows what it actually did, silently:

```
INFO  Creating replication slot with command CREATE_REPLICATION_SLOT "dbz_slot" LOGICAL pgoutput
INFO  Obtained valid replication slot ReplicationSlot [..., latestFlushedLsn=LSN{0/352B598}, ...]
```

It recreated `dbz_slot` from scratch at the **current** LSN. No error, no failed
task, no alert. The changes committed between the switchover and that recreation
are simply skipped.

## 5. Quantify the silent gap

```bash
# stop the workload first, then:
./scripts/gaps.sh
```

```
table rows (distinct n)         : 255
kafka distinct n                : 250
kafka raw messages              : 250  (duplicates: 0)
MISSING (in table, not in Kafka): 5
RESULT: DATA LOSS — 5 committed rows never reached Kafka (the silent gap)

missing n values: 100 101 102 103 104
```

Five committed rows (`n = 100..104`), exactly the window around the switchover,
**never reached Kafka**, and nothing reported a problem. That is the difference
between this scenario and [scenario 02](02-failover-slots.md): same cluster, same
switchover, one boolean on the slot.
