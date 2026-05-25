# Scenario 02 — Failover slots: the slot survives the switchover

**Claim proven (and verified on a real run):** a logical slot created with
`failover = true`, on a PG18 Patroni cluster with slot synchronization enabled, is
present on the new primary after a switchover. Debezium reconnects and resumes with
**zero loss** (a handful of duplicates, because CDC is at-least-once).

> Output below is from an actual run (lightly trimmed). Leader was `patroni2`,
> standby `patroni1` in this run; roles are dynamic, the scripts follow them.

## 1. Bring up + enable the failover machinery

```bash
./scripts/up.sh
./scripts/enable-failover-config.sh    # sets synchronized_standby_slots on the leader
./scripts/preflight.sh                  # expect all [PASS]
```

`preflight.sh` confirms the often-missed prerequisite, that `primary_conninfo`
contains a `dbname` (Patroni 4.1.3 injects it):

```
primary_conninfo  dbname=postgres user=replicator ... host=patroni2 ...
[PASS] primary_conninfo carries dbname (the slot-sync worker can connect)
```

## 2. Workload + failover connector

```bash
./scripts/workload.sh            # terminal B
./scripts/register-debezium.sh 02    # slot.name=cdc_slot, slot.failover=true  -> HTTP 201
```

## 3. The slot is synchronized to the standby

```bash
./scripts/slots.sh
```

```
===== patroni2 (LEADER) =====
 slot_name | slot_type | active | failover | synced | temporary | confirmed_flush_lsn
-----------+-----------+--------+----------+--------+-----------+---------------------
 cdc_slot  | logical   | t      | t        | f      | f         | 0/34B0A10
 patroni1  | physical  | t      | f        | f      | f         |

===== patroni1 (replica) =====
 slot_name | slot_type | active | failover | synced | temporary | confirmed_flush_lsn
-----------+-----------+--------+----------+--------+-----------+---------------------
 cdc_slot  | logical   | t      | t        | t      | f         | 0/34B0A10
```

On the standby, `cdc_slot` shows `synced=t`, `failover=t`, `temporary=f` and a
`confirmed_flush_lsn` that tracks the leader. That is the slot being mirrored.

### Important: wait for the synced slot to PERSIST before you trust it

A freshly synced slot is created `temporary=t` and is dropped on promotion until it
is persisted. On this run the sync worker first refused, logged on the standby:

```
LOG:  could not synchronize replication slot "cdc_slot"
DETAIL: Synchronization could lead to data loss, because the remote slot needs WAL
        at LSN 0/3443E00 and catalog xmin 760, but the standby has ... xmin 761.
```

then, once the primary slot advanced (Debezium flushes offsets every 10s in this
lab, see `OFFSET_FLUSH_INTERVAL_MS`), it persisted:

```
LOG:  newly created replication slot "cdc_slot" is sync-ready now
```

Only after `temporary=f` (sync-ready) is the slot safe across a failover. `preflight.sh`
and `slots.sh` let you confirm it.

## 4. Switch over (workload still running)

```bash
./scripts/switchover.sh
./scripts/enable-failover-config.sh   # the correct synchronized_standby_slots value flips
```

```
2026-... Successfully switched over to "patroni1"
[cfg] leader=patroni1  standby=patroni2  ->  synchronized_standby_slots='patroni2'
```

## 5. The slot survived; Debezium resumed

```bash
./scripts/slots.sh ; ./scripts/connect-status.sh orders-failover
```

```
===== patroni1 (LEADER) =====
 cdc_slot | logical | t | t | f | f | 0/34DEF58     <- now live on the new primary
{"conn":"RUNNING","task":"RUNNING"}
```

Kafka offsets kept advancing straight through the switchover (446 -> 556 in this run).

## 6. Prove zero loss

```bash
# stop the workload first, then:
./scripts/gaps.sh
```

```
table rows (distinct n)         : 616
kafka distinct n                : 616
kafka raw messages              : 690  (duplicates: 74)
MISSING (in table, not in Kafka): 0
RESULT: no loss — every committed row reached Kafka (duplicates expected; CDC is at-least-once)
```

**616 committed rows, 616 distinct events in Kafka, zero missing.** The 74 duplicates
are normal at-least-once redelivery around the reconnect, so the downstream consumer
must be idempotent, but nothing was lost. Compare with
[scenario 01](01-baseline-no-failover.md).
