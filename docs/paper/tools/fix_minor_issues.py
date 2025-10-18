#!/usr/bin/env python3
"""
Fix minor LaTeX issues - mathematical expressions, Unicode characters, and formatting.
"""
import re
from pathlib import Path

def fix_minor_issues():
    """Fix all minor LaTeX compilation issues."""
    
    base_dir = Path(__file__).resolve().parents[1]  # docs/paper
    with open(base_dir / 'paper_final.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Unicode character σ (U+03C3) - replace with \sigma
    content = content.replace('σ', '\\sigma')
    content = content.replace('π', '\\pi')
    content = content.replace('α', '\\alpha')
    content = content.replace('β', '\\beta')
    content = content.replace('γ', '\\gamma')
    content = content.replace('δ', '\\delta')
    content = content.replace('ε', '\\epsilon')
    content = content.replace('θ', '\\theta')
    content = content.replace('λ', '\\lambda')
    content = content.replace('μ', '\\mu')
    content = content.replace('ρ', '\\rho')
    content = content.replace('τ', '\\tau')
    content = content.replace('φ', '\\phi')
    content = content.replace('ω', '\\omega')
    
    # Fix 2: Bad math environment delimiters
    # Fix equation environments that are improperly nested or terminated
    content = re.sub(r'\\begin{equation}\s*\\begin{equation}', r'\\begin{equation}', content)
    content = re.sub(r'\\end{equation}\s*\\end{equation}', r'\\end{equation}', content)
    
    # Fix 3: Missing $ for math expressions
    # Fix inline math that should be in $ $ mode
    content = re.sub(r'([^\\])(\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+)([^$\\])', r'\1$\2$\3', content)
    
    # Fix 4: Display math issues - convert $$ to equation environment
    content = re.sub(r'\$\$([^$]+)\$\$', r'\\begin{equation}\\n\\1\\n\\end{equation}', content)
    
    # Fix 5: Fix bad \eqno usage in vertical mode
    content = re.sub(r'\\eqno', r'', content)
    
    # Fix 6: Clean up extra \endgroup
    content = re.sub(r'\\endgroup(?![\w])', r'', content)
    
    # Fix 7: Fix mathematical expressions that need proper math mode
    # Pattern: text followed by mathematical notation
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Skip lines that are already in equation environments or contain $
        if '\\begin{equation}' in line or '\\end{equation}' in line or '$' in line:
            fixed_lines.append(line)
            continue
            
        # Check for mathematical expressions that need $ wrapping
        if re.search(r'[a-zA-Z]\s*[_^]\s*[{a-zA-Z]', line):
            # This line contains subscripts/superscripts that need math mode
            line = re.sub(r'([a-zA-Z_]+)([_^]\{[^}]*\})', r'$\1\2$', line)
        
        # Check for Greek letters in text
        if re.search(r'\\(alpha|beta|gamma|delta|epsilon|sigma|pi|theta|lambda|mu|rho|tau|phi|omega)', line):
            if not line.strip().startswith('\\') and not '$' in line:
                # Wrap the whole line if it contains Greek letters outside of commands
                line = re.sub(r'\\([a-zA-Z]+)', r'$\\\1$', line)
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix 8: Remove any remaining problematic characters
    content = re.sub(r'[^\x00-\x7F\n\r\t]', '', content)  # Remove non-ASCII except newlines and tabs
    
    # Fix 9: Clean up multiple consecutive empty lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Fix 10: Ensure proper document structure
    if '\\documentclass' not in content:
        # Add minimal document structure if missing
        content = '\\documentclass[11pt]{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{amsmath,amsfonts,amssymb}\n\\usepackage{geometry}\n\\geometry{margin=1in}\n\\begin{document}\n\n' + content + '\n\n\\end{document}'
    
    return content

def main():
    """Apply all minor issue fixes."""
    print("Fixing minor LaTeX issues...")
    
    try:
        fixed_content = fix_minor_issues()
        
        # Write the clean version
        base_dir = Path(__file__).resolve().parents[1]
        with open(base_dir / 'paper_clean.tex', 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Created clean version: paper_clean.tex")
        print(f"File size: {len(fixed_content)} characters")
        
    except Exception as e:
        print(f"Error fixing issues: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
