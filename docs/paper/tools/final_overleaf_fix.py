#!/usr/bin/env python3
"""
Final fix for Overleaf compatibility - conservative approach.
"""
import re
from pathlib import Path

def create_safe_overleaf_version():
    """Create ultra-safe version for Overleaf."""
    
    base_dir = Path(__file__).resolve().parents[1]  # docs/paper
    with open(base_dir / 'paper_clean_v2.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Remove ALL non-standard Unicode characters aggressively
    # Keep only ASCII printable characters + essential LaTeX
    content = ''.join(char for char in content if ord(char) < 128 or char in '\n\r\t')
    
    # 2. Fix escaped underscores in math mode
    content = re.sub(r'\\\\_', r'_', content)
    
    # 3. Remove any remaining problematic constructs
    content = re.sub(r'```[a-zA-Z]*\n?', '', content)
    content = re.sub(r'style[^\\n]*', '', content)
    content = re.sub(r'class[^\\n]*', '', content)
    
    # 4. Fix display math
    content = re.sub(r'\$\$(.*?)\$\$', r'\\[\n\\1\n\\]', content, flags=re.DOTALL)
    
    # 5. Clean whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 6. Ensure clean preamble
    if not content.startswith('\\documentclass'):
        preamble = """\\documentclass[11pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{geometry}
\\geometry{margin=1in}

"""
        doc_start = content.find('\\begin{document}')
        if doc_start > -1:
            content = preamble + content[doc_start:]
    
    return content

def main():
    """Create final safe Overleaf version."""
    print("Creating ultra-safe Overleaf version...")
    
    try:
        safe_content = create_safe_overleaf_version()
        
        base_dir = Path(__file__).resolve().parents[1]
        with open(base_dir / 'paper_overleaf_safe.tex', 'w', encoding='utf-8') as f:
            f.write(safe_content)
        
        print(f"Created safe Overleaf version: paper_overleaf_safe.tex")
        print(f"Size: {len(safe_content)} characters")
        
        # Check structure
        open_braces = safe_content.count('{')
        close_braces = safe_content.count('}')
        print(f"Braces: {open_braces} open, {close_braces} close")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
