"""What is the lab doing right now? Embedding versions, the running model job, stale vectors.

    python python/status.py            # once
    watch -n 15 ~/venvs/lab16/bin/python python/status.py   # follow a background step
"""
import fcntl
import subprocess

from _common import LOCK_FILE, admin_conn


def main():
    with admin_conn().cursor() as cur:
        cur.execute("SELECT count(*) FROM bank.documents")
        total = cur.fetchone()[0]
        cur.execute("SELECT v.version_id, v.state, v.is_active, v.key_version, count(e.doc_id) "
                    "FROM bank.embedding_versions v LEFT JOIN bank.embeddings e USING (version_id) "
                    "GROUP BY v.version_id ORDER BY v.version_id")
        print("embedding versions")
        for vid, state, active, kv, n in cur.fetchall():
            flag = "active" if active else ("building" if n < total else "inactive")
            print(f"  {vid:>3}  {state:<10} {n:>5}/{total}  {flag:<8} key_version={kv}")
        cur.execute("SELECT state, count(*) FILTER (WHERE NOT fresh) FROM gov.embedding_staleness GROUP BY state")
        stale = {s: n for s, n in cur.fetchall() if n}
        print(f"stale active vectors: {stale or 'none'}")

    with open(LOCK_FILE, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
            print("model job: none running, you can start the next Python step")
        except BlockingIOError:
            f.seek(0)
            print(f"model job: RUNNING ({f.read().strip()}), wait before starting another Python step")
    jobs = subprocess.run(["pgrep", "-af", r"python/(embed|measure|evaluate_answers|agent)\.py"],
                          capture_output=True, text=True).stdout.strip()
    if jobs:
        print("processes:\n  " + jobs.replace("\n", "\n  "))


if __name__ == "__main__":
    main()
