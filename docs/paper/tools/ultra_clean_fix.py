#!/usr/bin/env python3
"""
Ultra-clean LaTeX fix specifically for Overleaf strict compilation.
"""
import re
from pathlib import Path

def ultra_clean_for_overleaf():
    """Apply ultra-aggressive cleaning for Overleaf."""
    
    base_dir = Path(__file__).resolve().parents[1]  # docs/paper
    with open(base_dir / 'paper_overleaf.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Remove ALL potential problematic characters
    # Keep only standard ASCII characters, LaTeX commands, and essential symbols
    content = re.sub(r'[^\x20-\x7E\n\r\t]', '', content)
    
    # Fix 2: Fix all mathematical expressions systematically
    # Ensure every math expression is properly formatted
    
    # Fix display math issues - remove any $$ constructs
    content = re.sub(r'\$\$(.*?)\$\$', r'\\[\n\\1\n\\]', content, flags=re.DOTALL)
    
    # Fix 3: Clean up equation environments
    # Ensure no empty equation environments
    content = re.sub(r'\\begin\{equation\}\s*\\end\{equation\}', '', content)
    
    # Fix 4: Handle subscripts and superscripts consistently
    # All text subscripts should use \\text{}
    content = re.sub(r'_([a-zA-Z]+)([^{])', r'_{\\text{\\1}}\\2', content)
    content = re.sub(r'_\{([a-zA-Z][a-zA-Z-]+)\}', r'_{\\text{\\1}}', content)
    
    # Fix 5: Clean up specific problem areas
    # Remove any remaining code block artifacts
    content = re.sub(r'style\s+[^\\n]*', '', content)
    content = re.sub(r'class\s+[^\\n]*', '', content)
    
    # Fix 6: Ensure proper spacing around math environments
    lines = content.split('\\n')
    cleaned_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines in sequences
        if not stripped:
            if i > 0 and not lines[i-1].strip():
                continue  # Skip consecutive empty lines
        
        # Clean up problematic line structures
        if stripped.startswith('\\begin{equation}') or stripped.startswith('\\end{equation}'):
            cleaned_lines.append('')  # Add spacing
            cleaned_lines.append(line)
            cleaned_lines.append('')  # Add spacing
        else:
            cleaned_lines.append(line)
    
    content = '\\n'.join(cleaned_lines)
    
    # Fix 7: Handle inline math more carefully
    def fix_inline_math(match):
        math_content = match.group(1)
        # Ensure no problematic characters in inline math
        math_content = re.sub(r'\\\\_', '_', math_content)
        return f'${math_content}$'
    
    content = re.sub(r'\\$([^$]+)\\$', fix_inline_math, content)
    
    # Fix 8: Clean up any remaining escape issues
    content = re.sub(r'\\\\+', '\\\\', content)  # Remove excessive backslashes
    content = re.sub(r'\\\\_', '_', content)      # Fix escaped underscores
    
    # Fix 9: Ensure proper document structure
    # Add minimal but complete preamble
    preamble = """\\documentclass[11pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{geometry}
\\geometry{margin=1in}

"""
    
    # Replace the preamble section
    doc_start = content.find('\\begin{document}')
    if doc_start > -1:
        doc_content = content[doc_start:]
        content = preamble + doc_content
    
    # Fix 10: Final cleanup
    # Remove any remaining problematic constructs
    content = re.sub(r'\\\\n', '\\n', content)  # Fix double newlines
    content = re.sub(r'\\n{3,}', '\\n\\n', content)  # Limit consecutive newlines
    
    # Fix 11: Ensure all brackets are balanced
    # This is a simple check - could be made more sophisticated
    open_braces = content.count('{')
    close_braces = content.count('}')
    print(f"Brace count: {open_braces} open, {close_braces} close")
    
    return content

def main():
    """Create ultra-clean version for Overleaf."""
    print("Creating ultra-clean LaTeX for Overleaf...")
    
    try:
        ultra_clean_content = ultra_clean_for_overleaf()
        
        # Write the ultra-clean version
        base_dir = Path(__file__).resolve().parents[1]
        with open(base_dir / 'paper_ultra_clean.tex', 'w', encoding='utf-8') as f:
            f.write(ultra_clean_content)
        
        print(f"Created ultra-clean version: paper_ultra_clean.tex")
        print(f"File size: {len(ultra_clean_content)} characters")
        
    except Exception as e:
        print(f"Error creating ultra-clean version: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
