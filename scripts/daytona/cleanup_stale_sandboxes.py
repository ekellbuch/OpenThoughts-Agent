#!/usr/bin/env python3
"""
Delete Daytona sandboxes in the RL org that have not had an event in over an hour.

Uses the official Daytona Python SDK (`daytona` package, v0.180+) to list all
active sandboxes and delete any whose `lastActivityAt` (was: `updatedAt`)
timestamp is older than the configured threshold.

History:
    * 2026-05-18 — switched from the deprecated GET /api/sandbox/paginated
      (offset pagination) to GET /api/sandbox (cursor pagination) ahead of
      Daytona's 2026-05-24 breaking change.
    * 2026-05-28 — migrated from raw `requests` to the `daytona` SDK after the
      hand-rolled query string started returning HTTP 400. Three things had
      changed server-side post-cutover:
        1. `states` is now a *multi*-value query param (repeated
           `?states=started`), not a single comma-separated string.
        2. `sort=updatedAt` is no longer a valid value — the new enum is
           `name | cpu | memoryGib | diskGib | lastActivityAt | createdAt`
           (use `lastActivityAt` to get the old "most-recently-active first"
           ordering).
        3. The `updatedAt` response field still exists for backward
           compatibility, but `lastActivityAt` is what the freshness check
           should hang off of going forward. We prefer `last_activity_at` and
           fall back to `updated_at` for any pre-cutover hosts.
      Using the SDK insulates us from future server-side wire changes — the
      SDK is regenerated from Daytona's OpenAPI spec and tracks the API.

Usage:
    # Dry run (default) — shows what would be deleted
    python cleanup_stale_sandboxes.py

    # Actually delete
    python cleanup_stale_sandboxes.py --delete

    # Custom threshold (minutes)
    python cleanup_stale_sandboxes.py --delete --threshold 120

Environment:
    DAYTONA_API_KEY     — API key for the target organization (default)
    DAYTONA_RL_API_KEY  — Fallback API key (or set in secrets.env)
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from daytona import (
    Daytona,
    DaytonaConfig,
    ListSandboxesQuery,
    SandboxListSortDirection,
    SandboxListSortField,
    SandboxState,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PAGE_LIMIT = 100  # per-page fetch size; SDK pages through automatically
DEFAULT_THRESHOLD_MINUTES = 60

# Try to load secrets.env from a few common locations
SECRET_ENV_PATH = os.environ.get("DC_AGENT_SECRET_ENV")
if SECRET_ENV_PATH and os.path.isfile(SECRET_ENV_PATH):
    load_dotenv(SECRET_ENV_PATH)
else:
    # Fallback: look next to this script or in ~/Documents
    for candidate in [
        os.path.join(os.path.dirname(__file__), "..", "..", "secrets.env"),
        os.path.expanduser("~/Documents/secrets.env"),
    ]:
        if os.path.isfile(candidate):
            load_dotenv(candidate)
            break


def get_api_key(env_var: str = "DAYTONA_API_KEY") -> str:
    key = os.environ.get(env_var)
    if not key:
        # Fallback chain: try common key names
        for fallback in ("DAYTONA_API_KEY", "DAYTONA_RL_API_KEY"):
            if fallback != env_var:
                key = os.environ.get(fallback)
                if key:
                    break
    if not key:
        sys.exit(f"ERROR: {env_var} (and fallbacks) not set in environment or secrets.env")
    return key


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _last_seen_ts(sb) -> str | None:
    """Pick the freshest activity timestamp available on a Sandbox object.

    Prefers `last_activity_at` (post-cutover canonical), falls back to
    `updated_at` (pre-cutover field still present on the response).
    """
    return getattr(sb, "last_activity_at", None) or getattr(sb, "updated_at", None)


def list_started_sandboxes(client: Daytona) -> list:
    """Fetch all sandboxes in 'started' state, sorted by last activity desc.

    The SDK transparently handles cursor pagination; we just iterate the
    generator.
    """
    query = ListSandboxesQuery(
        states=[SandboxState.STARTED],
        sort=SandboxListSortField.LASTACTIVITYAT,
        order=SandboxListSortDirection.DESC,
        limit=PAGE_LIMIT,
    )
    return list(client.list(query))


def delete_sandbox(sb) -> bool:
    """Delete a single sandbox. Returns True on success."""
    try:
        sb.delete()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    error: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_stale_sandboxes(sandboxes: list, threshold_minutes: int) -> list:
    """Return sandboxes whose last-activity timestamp is older than threshold_minutes ago."""
    now = datetime.now(timezone.utc)
    stale = []

    for sb in sandboxes:
        updated_str = _last_seen_ts(sb)
        if not updated_str:
            continue
        # Parse ISO-8601 timestamp (with or without trailing Z)
        updated_str = updated_str.replace("Z", "+00:00")
        try:
            updated_at = datetime.fromisoformat(updated_str)
        except ValueError:
            continue
        age_minutes = (now - updated_at).total_seconds() / 60.0
        if age_minutes > threshold_minutes:
            # Attach for the report; using a plain attribute since these are
            # SDK Sandbox instances, not dicts.
            sb._age_minutes = round(age_minutes, 1)
            stale.append(sb)

    return stale


def main():
    parser = argparse.ArgumentParser(
        description="Clean up stale Daytona RL sandboxes"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete stale sandboxes (default is dry-run)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD_MINUTES,
        help=f"Minutes of inactivity before a sandbox is considered stale (default: {DEFAULT_THRESHOLD_MINUTES})",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="DAYTONA_API_KEY",
        help="Environment variable name containing the API key (default: DAYTONA_API_KEY)",
    )
    args = parser.parse_args()

    api_key = get_api_key(args.api_key_env)
    client = Daytona(DaytonaConfig(api_key=api_key))

    # 1. List all active sandboxes
    print("Fetching all started sandboxes …")
    sandboxes = list_started_sandboxes(client)
    print(f"  Found {len(sandboxes)} started sandboxes.")

    # 2. Find stale ones
    stale = find_stale_sandboxes(sandboxes, args.threshold)
    print(f"  {len(stale)} are stale (no event in >{args.threshold} min).\n")

    if not stale:
        print("Nothing to clean up.")
        return

    # 3. Print summary
    print(f"{'ID':<40} {'Age (min)':>10}  {'Created':<26} {'Last Activity':<26}")
    print("-" * 110)
    for sb in stale:
        created = sb.created_at or "n/a"
        last_seen = _last_seen_ts(sb) or "n/a"
        print(
            f"{sb.id:<40} {sb._age_minutes:>10.1f}  "
            f"{created:<26} {last_seen:<26}"
        )

    if not args.delete:
        print(f"\nDry run — pass --delete to actually remove these {len(stale)} sandboxes.")
        return

    # 4. Delete
    print(f"\nDeleting {len(stale)} stale sandboxes …")
    success = 0
    failed = 0

    for i, sb in enumerate(stale, 1):
        ok = delete_sandbox(sb)
        if ok:
            success += 1
        else:
            failed += 1
            print(f"  FAILED to delete {sb.id}")

        # Progress every 50
        if i % 50 == 0 or i == len(stale):
            print(f"  Progress: {i}/{len(stale)}  (ok={success}, fail={failed})")

        # Small delay to avoid rate-limiting
        time.sleep(0.05)

    print(f"\nDone. Deleted {success}/{len(stale)} sandboxes.")
    if failed:
        print(f"  {failed} deletions failed.")


if __name__ == "__main__":
    main()
