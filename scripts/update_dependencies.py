#!/usr/bin/env python3
"""Update vulnerable dependencies to their latest secure versions."""

import re
import sys
from pathlib import Path

# Define vulnerable packages and their recommended versions
SECURITY_UPDATES = {
    # Based on common CVEs and GitHub security advisories
    'aiohttp': '3.10.11->3.11.18',  # Update to latest 3.11.x for compatibility
    'tornado': '6.4.2->6.4.2',  # Already on latest 6.4.x
    'cryptography': '44.0.2->44.0.2',  # Already on latest
    'urllib3': '2.4.0->2.4.0',  # Already on latest
    'werkzeug': None,  # Check if present
    'jinja2': None,  # Check if present
    'flask': '3.0.3->3.0.3',  # Already on latest
    'pillow': '11.2.1->11.2.1',  # Already on latest
    'requests': '2.32.3->2.32.3',  # Already on latest
}

def update_requirements_file(file_path: Path):
    """Update requirements.txt with security patches."""
    if not file_path.exists():
        print(f"File {file_path} not found!")
        return
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    updated_lines = []
    updates_made = []
    
    for line in lines:
        updated_line = line
        
        # Skip empty lines and comments
        if line.strip() == '' or line.strip().startswith('#'):
            updated_lines.append(line)
            continue
            
        # Parse package name and version
        match = re.match(r'^([a-zA-Z0-9-_.]+)==(.+)$', line.strip())
        if match:
            package_name = match.group(1).lower()
            current_version = match.group(2)
            
            if package_name in SECURITY_UPDATES and SECURITY_UPDATES[package_name]:
                old_ver, new_ver = SECURITY_UPDATES[package_name].split('->')
                if current_version == old_ver and old_ver != new_ver:
                    updated_line = f"{package_name}=={new_ver}\n"
                    updates_made.append(f"{package_name}: {old_ver} -> {new_ver}")
        
        updated_lines.append(updated_line)
    
    if updates_made:
        print(f"\nUpdating {file_path}:")
        for update in updates_made:
            print(f"  - {update}")
        
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)
    else:
        print(f"\nNo security updates needed for {file_path}")

def update_pyproject_toml(file_path: Path):
    """Update pyproject.toml with security patches."""
    if not file_path.exists():
        print(f"File {file_path} not found!")
        return
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    updates_made = []
    
    # Update aiohttp to latest 3.11.x for security fixes
    if 'aiohttp' in content:
        # Update to use caret notation for minor version updates
        content = re.sub(
            r'aiohttp\s*=\s*"[^"]+"',
            'aiohttp = "^3.11.18"',
            content
        )
        updates_made.append("aiohttp: updated to ^3.11.18")
    
    # Update cryptography if needed
    if 'cryptography = "^42.0.0"' in content:
        content = content.replace(
            'cryptography = "^42.0.0"',
            'cryptography = "^44.0.0"'
        )
        updates_made.append("cryptography: ^42.0.0 -> ^44.0.0")
    
    # Update version number for next release (4-digit versioning)
    content = re.sub(
        r'version\s*=\s*"1\.7\.0"',
        'version = "1.8.0.0"',
        content
    )
    updates_made.append("version: 1.7.0 -> 1.8.0.0 (switching to 4-digit versioning)")
    
    if updates_made:
        print(f"\nUpdating {file_path}:")
        for update in updates_made:
            print(f"  - {update}")
        
        with open(file_path, 'w') as f:
            f.write(content)
    else:
        print(f"\nNo updates needed for {file_path}")

def main():
    """Main function to update all dependency files."""
    print("Security Dependency Update Tool")
    print("=" * 40)
    
    # Update requirements.txt
    requirements_file = Path("requirements.txt")
    update_requirements_file(requirements_file)
    
    # Update pyproject.toml
    pyproject_file = Path("pyproject.toml")
    update_pyproject_toml(pyproject_file)
    
    print("\n" + "=" * 40)
    print("Security update check complete!")
    print("\nNext steps:")
    print("1. Run 'pip install -r requirements.txt' to update packages")
    print("2. Run 'poetry update' if using Poetry")
    print("3. Run tests to ensure everything works")
    print("4. Commit the changes")

if __name__ == "__main__":
    main()