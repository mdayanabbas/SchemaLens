#!/usr/bin/env python3
import argparse
import asyncio
import sys
from pathlib import Path

# Add backend to path so we can import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.audit.service import AuditService
from app.core.config import get_settings
from app.db.session import async_session_factory
from app.workflows.schema_scan_recovery import SchemaScanRecoveryService


async def main():
    parser = argparse.ArgumentParser(description="Recover stale schema scans.")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=100,
        help="Maximum number of scans to process (default: 100)"
    )
    args = parser.parse_args()

    if args.limit <= 0 or args.limit > 1000:
        print("Error: Limit must be between 1 and 1000.", file=sys.stderr)
        sys.exit(1)

    # Validate settings can load
    get_settings()

    try:
        async with async_session_factory() as session:
            audit_service = AuditService(session)
            recovery_service = SchemaScanRecoveryService(session, audit_service)
            
            result = await recovery_service.recover_stale_scans(limit=args.limit)
            
            print(f"Inspected: {result.inspected}")
            print(f"Failed: {result.failed}")
            print(f"Cancelled: {result.cancelled}")
            
    except Exception as e:
        print(f"Error during recovery: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
