# from pymilvus import connections, utility
# from app.core.config import settings

# # Connect to Milvus
# connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

# # List all existing collections
# collections = utility.list_collections()
# print("📋 Existing collections:", collections)

# # Drop each collection
# for name in collections:
#     utility.drop_collection(name)
#     print(f"🗑️ Dropped collection: {name}")

# print("✅ All collections deleted successfully!")
#!/usr/bin/env python3


import os
import argparse
import re
import logging
import sys
from pathlib import Path

# When running this script directly (python app/utils/drop_all_collections.py)
# the package root may not be on sys.path, which causes `ModuleNotFoundError: No module named 'app'`.
# Add the repository root to sys.path so `from app.core.config import settings` works.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymilvus import connections, utility
from app.core.config import settings

logging.basicConfig(filename='drop_collections.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def parse_args():
    p = argparse.ArgumentParser(description="Safely drop Milvus collections.")
    p.add_argument("--dry-run", action="store_true", help="Print collections that would be dropped and exit.")
    p.add_argument("--filter", type=str, default=".*", help="Regex to match collection names (default: all).")
    p.add_argument("--yes", action="store_true", help="Skip interactive confirmation (use with care).")
    p.add_argument("--require-env", type=str, default="ALLOW_DROP_ALL", help="Require this env var to be '1' to allow drops.")
    return p.parse_args()

def main():
    args = parse_args()
    if os.environ.get(args.require_env, "0") != "1":
        print(f"Environment guard: set {args.require_env}=1 to allow destructive operations.")
        return

    connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    all_cols = utility.list_collections()
    pattern = re.compile(args.filter)
    to_drop = [c for c in all_cols if pattern.match(c)]

    if not to_drop:
        print("No collections match the filter. Nothing to do.")
        return

    print("Collections matching filter:")
    for c in to_drop:
        print(" -", c)

    if args.dry_run:
        print("Dry run mode; no collections will be dropped.")
        return

    if not args.yes:
        confirm = input("Type 'YES' to permanently drop these collections: ")
        if confirm != "YES":
            print("Aborted by user.")
            return

    for name in to_drop:
        try:
            utility.drop_collection(name)
            logging.info("Dropped collection: %s", name)
            print("Dropped:", name)
        except Exception as e:
            logging.exception("Failed to drop %s: %s", name, e)
            print("Failed to drop", name, ":", e)

    print("Done. See drop_collections.log for details.")

if __name__ == "__main__":
    main()