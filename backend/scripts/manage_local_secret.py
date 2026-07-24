import argparse
import base64
import secrets
import sys

def generate_key() -> None:
    """Generate a secure 32-byte master key and output environment configuration."""
    raw_key = secrets.token_bytes(32)
    encoded_key = base64.b64encode(raw_key).decode('utf-8')
    
    print("\nLocal Secret Master Key successfully generated.\n")
    print("Add the following configuration to your environment or .env file:")
    print("-" * 60)
    print(f"LOCAL_SECRET_MASTER_KEY={encoded_key}")
    print("LOCAL_SECRET_KEY_VERSION=v1")
    print("-" * 60)
    print("\nWARNING: Keep this key secure and never commit it to version control.")
    print("If this key is lost, all locally encrypted secrets will be permanently unrecoverable.\n")

def main():
    parser = argparse.ArgumentParser(description="Manage SchemaLens local secrets configuration.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("generate-key", help="Generate a new local secret master key.")
    
    args = parser.parse_args()
    
    if args.command == "generate-key":
        generate_key()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
