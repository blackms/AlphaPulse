#!/usr/bin/env python3
"""
Migration script to move from hardcoded credentials to secure secret management.

This script helps migrate existing credentials to the new secure system.
"""
import os
import sys
import json
import argparse
import getpass
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpha_pulse.utils.secrets_manager import (
    create_secrets_manager,
    LocalEncryptedFileProvider,
    EnvironmentSecretProvider
)


def load_old_credentials() -> Dict[str, Any]:
    """Load existing hardcoded credentials from known locations."""
    credentials = {}
    
    # Check for old credential files
    old_cred_paths = [
        "src/alpha_pulse/exchanges/credentials/binance_credentials.json",
        "src/alpha_pulse/exchanges/credentials/bybit_credentials.json",
        ".env"
    ]
    
    for path in old_cred_paths:
        if os.path.exists(path):
            print(f"Found old credential file: {path}")
            
            if path.endswith(".json"):
                with open(path, 'r') as f:
                    data = json.load(f)
                    exchange = Path(path).stem.replace("_credentials", "")
                    credentials[f"{exchange}_credentials"] = data
            
            elif path == ".env":
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            credentials[key.lower()] = value.strip('"').strip("'")
    
    return credentials


def create_env_template() -> str:
    """Create a template .env file with all required variables."""
    template = """# AlphaPulse Environment Configuration
# Copy this file to .env and fill in your values

# Environment
ALPHAPULSE_ENV=development

# Database Configuration
ALPHAPULSE_DB_HOST=localhost
ALPHAPULSE_DB_PORT=5432
ALPHAPULSE_DB_USER=your_db_user
ALPHAPULSE_DB_PASSWORD=your_db_password
ALPHAPULSE_DB_NAME=alphapulse

# Security Configuration
ALPHAPULSE_JWT_SECRET=your-very-long-random-string-here
ALPHAPULSE_ENCRYPTION_KEY=your-32-byte-encryption-key-here

# Exchange API Credentials
# Binance
ALPHAPULSE_BINANCE_API_KEY=your_binance_api_key
ALPHAPULSE_BINANCE_API_SECRET=your_binance_api_secret

# Bybit
ALPHAPULSE_BYBIT_API_KEY=your_bybit_api_key
ALPHAPULSE_BYBIT_API_SECRET=your_bybit_api_secret

# Data Provider API Keys
ALPHAPULSE_IEX_CLOUD_API_KEY=your_iex_cloud_key
ALPHAPULSE_POLYGON_API_KEY=your_polygon_key
ALPHAPULSE_ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
ALPHAPULSE_FINNHUB_API_KEY=your_finnhub_key

# AWS Configuration (for production)
# AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key

# Vault Configuration (for staging)
# VAULT_URL=http://localhost:8200
# VAULT_TOKEN=your_vault_token
"""
    return template


def migrate_to_env_file(credentials: Dict[str, Any], output_path: str = ".env.secure"):
    """Migrate credentials to environment file format."""
    env_lines = []
    
    # Map old keys to new environment variables
    key_mapping = {
        "binance_credentials": {
            "api_key": "ALPHAPULSE_BINANCE_API_KEY",
            "api_secret": "ALPHAPULSE_BINANCE_API_SECRET"
        },
        "bybit_credentials": {
            "api_key": "ALPHAPULSE_BYBIT_API_KEY",
            "api_secret": "ALPHAPULSE_BYBIT_API_SECRET"
        },
        "alpha_pulse_bybit_api_key": "ALPHAPULSE_BYBIT_API_KEY",
        "alpha_pulse_bybit_api_secret": "ALPHAPULSE_BYBIT_API_SECRET",
        "jwt_secret": "ALPHAPULSE_JWT_SECRET"
    }
    
    # Process credentials
    for key, value in credentials.items():
        if key in key_mapping:
            if isinstance(value, dict):
                # Handle nested credentials
                for sub_key, env_var in key_mapping[key].items():
                    if sub_key in value:
                        env_lines.append(f"{env_var}={value[sub_key]}")
            else:
                # Direct mapping
                env_var = key_mapping.get(key, f"ALPHAPULSE_{key.upper()}")
                env_lines.append(f"{env_var}={value}")
        else:
            # Default mapping
            env_var = f"ALPHAPULSE_{key.upper()}"
            if isinstance(value, dict):
                env_lines.append(f"# {key} (complex value - needs manual migration)")
                env_lines.append(f"# {env_var}={json.dumps(value)}")
            else:
                env_lines.append(f"{env_var}={value}")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("# Migrated AlphaPulse Credentials\n")
        f.write("# Review and update these values before using\n\n")
        f.write("\n".join(env_lines))
    
    print(f"✅ Migrated credentials written to: {output_path}")


def migrate_to_encrypted_files(credentials: Dict[str, Any], secrets_dir: str = ".secrets"):
    """Migrate credentials to encrypted local files."""
    print("\n🔐 Setting up encrypted local storage...")
    
    # Get or generate encryption key
    encryption_key = os.environ.get("ALPHAPULSE_ENCRYPTION_KEY")
    if not encryption_key:
        print("No encryption key found. Generating a new one...")
        from cryptography.fernet import Fernet
        encryption_key = Fernet.generate_key().decode()
        print(f"\n⚠️  IMPORTANT: Save this encryption key securely!")
        print(f"Encryption key: {encryption_key}")
        print("You'll need this key to access your secrets.\n")
    
    # Create provider
    provider = LocalEncryptedFileProvider(secrets_dir, encryption_key)
    
    # Store each credential
    for key, value in credentials.items():
        if provider.set_secret(key, value):
            print(f"✅ Stored: {key}")
        else:
            print(f"❌ Failed to store: {key}")
    
    print(f"\n✅ Secrets stored in: {secrets_dir}/")
    print("Remember to add this directory to .gitignore!")


def cleanup_old_files():
    """Remove old credential files after migration."""
    old_files = [
        "src/alpha_pulse/exchanges/credentials/binance_credentials.json",
        "src/alpha_pulse/exchanges/credentials/bybit_credentials.json"
    ]
    
    print("\n🧹 Cleaning up old credential files...")
    
    for file_path in old_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✅ Removed: {file_path}")
            except Exception as e:
                print(f"❌ Failed to remove {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate AlphaPulse credentials to secure storage"
    )
    parser.add_argument(
        "--method",
        choices=["env", "encrypted", "both"],
        default="env",
        help="Migration method (default: env)"
    )
    parser.add_argument(
        "--output",
        default=".env.secure",
        help="Output file for env method (default: .env.secure)"
    )
    parser.add_argument(
        "--secrets-dir",
        default=".secrets",
        help="Directory for encrypted files (default: .secrets)"
    )
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create a template .env file"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove old credential files after migration"
    )
    
    args = parser.parse_args()
    
    # Create template if requested
    if args.create_template:
        template = create_env_template()
        with open(".env.template", 'w') as f:
            f.write(template)
        print("✅ Created .env.template")
        return
    
    # Load existing credentials
    print("🔍 Searching for existing credentials...")
    credentials = load_old_credentials()
    
    if not credentials:
        print("❌ No existing credentials found to migrate")
        print("💡 Run with --create-template to create a template .env file")
        return
    
    print(f"\n📋 Found {len(credentials)} credential entries to migrate")
    
    # Perform migration
    if args.method in ["env", "both"]:
        print(f"\n📄 Migrating to environment file: {args.output}")
        migrate_to_env_file(credentials, args.output)
    
    if args.method in ["encrypted", "both"]:
        migrate_to_encrypted_files(credentials, args.secrets_dir)
    
    # Cleanup if requested
    if args.cleanup:
        confirm = input("\n⚠️  Delete old credential files? This cannot be undone! (yes/no): ")
        if confirm.lower() == "yes":
            cleanup_old_files()
        else:
            print("Skipping cleanup")
    
    # Final instructions
    print("\n✅ Migration complete!")
    print("\n📋 Next steps:")
    print("1. Review the migrated credentials")
    print("2. Update any missing or placeholder values")
    print("3. Add .secrets/ and .env.secure to .gitignore")
    print("4. Test the application with new credentials")
    print("5. Rotate any exposed API keys")
    
    if args.method == "env":
        print(f"\n💡 To use the migrated credentials:")
        print(f"   cp {args.output} .env")
        print(f"   # Edit .env to update any values")


if __name__ == "__main__":
    main()