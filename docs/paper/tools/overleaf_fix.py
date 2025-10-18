#!/usr/bin/env python3
"""
Create Overleaf-compatible LaTeX with perfect mathematical formula rendering.
"""
import re
from pathlib import Path

def create_overleaf_compatible():
    """Create ultra-clean LaTeX for Overleaf compatibility."""
    
    base_dir = Path(__file__).resolve().parents[1]  # docs/paper
    with open(base_dir / 'paper_clean_v2.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Remove problematic packages that might cause Overleaf issues
    content = re.sub(r'\\usepackage\{natbib\}', r'% \\usepackage{natbib} % Removed for Overleaf compatibility', content)
    
    # Fix 2: Ensure all mathematical expressions are properly enclosed
    # Fix standalone math expressions that need proper environments
    lines = content.split('\n')
    fixed_lines = []
    in_equation = False
    in_document = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if '\\begin{document}' in line:
            in_document = True
        
        if '\\begin{equation}' in line:
            in_equation = True
        elif '\\end{equation}' in line:
            in_equation = False
        
        # Skip header and preamble issues
        if not in_document:
            fixed_lines.append(line)
            continue
            
        # Fix mathematical expressions that are not in proper environments
        if not in_equation and in_document:
            # Check for mathematical content that needs wrapping
            if re.search(r'^[^%]*\\(tilde|hat|mathbf|sum|pi|sigma|alpha|beta|gamma|theta|lambda)', stripped):
                if not stripped.startswith('\\section') and not stripped.startswith('\\subsection'):
                    if not stripped.startswith('\\begin{') and not stripped.startswith('\\end{'):
                        if stripped and not stripped.startswith('%'):
                            # This looks like a mathematical expression
                            fixed_lines.append('\\begin{equation}')
                            fixed_lines.append(line)
                            fixed_lines.append('\\end{equation}')
                            continue
        
        # Fix inline math that might be problematic
        if '$' in line and not in_equation:
            # Count dollars to ensure they're balanced
            dollar_count = line.count('$')
            if dollar_count % 2 != 0:
                # Unbalanced dollars - fix them
                line = re.sub(r'\$([^$]*)\$([^$]*)\$', r'$\\1$ \\2', line)
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix 3: Clean up problematic math mode issues
    # Fix escaped underscores in math mode
    content = re.sub(r'\\begin\{equation\}(.*?)\\\\_', r'\\begin{equation}\\1_', content, flags=re.DOTALL)
    content = re.sub(r'\$(.*?)\\\\_', r'$\\1_', content, flags=re.DOTALL)
    
    # Fix 4: Ensure proper text mode in subscripts where needed
    content = re.sub(r'_\{([a-zA-Z]+)\}', r'_{\\text{\\1}}', content)
    content = re.sub(r'_\{([a-zA-Z]+-[a-zA-Z]+)\}', r'_{\\text{\\1}}', content)
    content = re.sub(r'_\{([a-zA-Z]+)-([a-zA-Z]+)-([a-zA-Z]+)\}', r'_{\\text{\\1-\\2-\\3}}', content)
    
    # Fix 5: Handle display math properly
    content = re.sub(r'\$\$(.*?)\$\$', r'\\[\n\\1\n\\]', content, flags=re.DOTALL)
    
    # Fix 6: Clean up any remaining problematic constructs
    # Remove any remaining triple backticks
    content = re.sub(r'```[a-zA-Z]*\n?', '', content)
    content = re.sub(r'```', '', content)
    
    # Fix 7: Ensure proper document structure with minimal packages
    preamble = """\\documentclass[11pt,a4paper]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{array}
\\usepackage{geometry}
\\usepackage{url}
\\usepackage{hyperref}
\\usepackage{longtable}

\\geometry{margin=1in}

"""
    
    # Find the document start
    doc_start = content.find('\\begin{document}')
    if doc_start > -1:
        doc_content = content[doc_start:]
        content = preamble + doc_content
    
    # Fix 8: Clean up excessive whitespace
    content = re.sub(r'\n{4,}', '\n\n', content)
    
    # Fix 9: Ensure equations are properly spaced
    content = re.sub(r'([^\n])\\begin{equation}', r'\\1\n\n\\begin{equation}', content)
    content = re.sub(r'\\end{equation}([^\n])', r'\\end{equation}\n\n\\1', content)
    
    # Fix 10: Fix specific problematic patterns for Overleaf
    # Clean up any remaining escape sequences that might cause issues
    content = re.sub(r'\\\\\_', r'_', content)
    content = re.sub(r'\\\\_', r'_', content)
    
    # Fix 11: Handle text in math mode properly
    content = re.sub(r'\{\\text\{([^}]*)\}([_^][^}]*\})\}', r'{\\text{\\1}\\2}', content)
    
    return content

def main():
    """Create Overleaf-compatible version."""
    print("Creating Overleaf-compatible LaTeX with perfect formula rendering...")
    
    try:
        overleaf_content = create_overleaf_compatible()
        
        # Write the Overleaf version
        base_dir = Path(__file__).resolve().parents[1]
        with open(base_dir / 'paper_overleaf.tex', 'w', encoding='utf-8') as f:
            f.write(overleaf_content)
        
        print(f"Created Overleaf version: paper_overleaf.tex")
        print(f"File size: {len(overleaf_content)} characters")
        
    except Exception as e:
        print(f"Error creating Overleaf version: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
