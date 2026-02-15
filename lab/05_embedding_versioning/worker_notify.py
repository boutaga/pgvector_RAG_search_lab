#!/usr/bin/env python3
"""
Lab 5 — Embedding Worker (LISTEN/NOTIFY variant)

Uses PostgreSQL LISTEN/NOTIFY to react instantly to new queue entries,
with a polling fallback for reliability.

Note: Uses select.select() for socket-level notification waiting.
      This works on Linux/macOS/WSL2 but not on native Windows.

Usage:
    python worker_notify.py
    python worker_notify.py --batch-size 10
"""

import argparse
import json
import logging
import os
import select
import signal
import sys
import time

import psycopg
from psycopg import sql

# Reuse processing logic from the polling worker
from worker import (
    DB_URL,
    DEFAULT_MODEL,
    OPENAI_API_KEY,
    POLL_INTERVAL,
    fetch_and_process,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NOTIFY] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

shutdown_requested = False


def handle_signal(signum, frame):
    global shutdown_requested
    log.info("Shutdown signal received")
    shutdown_requested = True


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def listen_loop(db_url: str, batch_size: int, model: str):
    """Listen for embedding_updates notifications and process on arrival."""
    log.info("Starting LISTEN/NOTIFY worker (model=%s)", model)

    # Use a dedicated connection for LISTEN (must be in autocommit mode)
    with psycopg.connect(db_url, autocommit=True) as listen_conn:
        listen_conn.execute("LISTEN embedding_updates")
        log.info("Listening on channel 'embedding_updates'")

        # Process any items already in queue on startup
        processed = fetch_and_process(db_url, batch_size, model, "notify-worker")
        if processed:
            log.info("Startup: processed %d queued items", processed)

        while not shutdown_requested:
            # Wait for notification with timeout (fallback polling)
            if select.select([listen_conn.fileno()], [], [], POLL_INTERVAL) == ([], [], []):
                # Timeout — poll anyway in case we missed a notification
                processed = fetch_and_process(db_url, batch_size, model, "notify-worker")
                if processed:
                    log.info("Poll fallback: processed %d items", processed)
                continue

            # Drain all pending notifications
            listen_conn.execute("SELECT 1")  # force read from socket
            notifications = []
            for notify in listen_conn.notifies():
                try:
                    payload = json.loads(notify.payload)
                    notifications.append(payload)
                    log.info(
                        "Notification: article_id=%s change_type=%s",
                        payload.get("article_id"),
                        payload.get("change_type"),
                    )
                except (json.JSONDecodeError, TypeError):
                    log.warning("Invalid notification payload: %s", notify.payload)

            if notifications:
                processed = fetch_and_process(db_url, batch_size, model, "notify-worker")
                log.info("Processed %d items after %d notifications", processed, len(notifications))

    log.info("LISTEN/NOTIFY worker stopped")


def main():
    parser = argparse.ArgumentParser(description="LISTEN/NOTIFY embedding worker")
    parser.add_argument("--batch-size", type=int, default=10, help="Items per batch")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="Database URL")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    listen_loop(args.db_url, args.batch_size, args.model)


if __name__ == "__main__":
    main()
