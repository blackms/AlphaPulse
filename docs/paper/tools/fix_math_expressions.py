#!/usr/bin/env python3
"""
Fix mathematical expressions in the comprehensive LaTeX paper.
"""
import re
from pathlib import Path

def fix_math_expressions():
    """Fix mathematical expressions for proper LaTeX compilation."""
    
    base_dir = Path(__file__).resolve().parents[1]  # docs/paper
    with open(base_dir / 'paper_comprehensive.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Remove remaining ```math blocks that weren't converted properly
    content = re.sub(r'```math\n(.*?)\n```', r'\\begin{equation}\n\1\n\\end{equation}', content, flags=re.DOTALL)
    
    # Fix 2: Convert math expressions mixed with text
    # Fix inline math mode issues
    content = re.sub(r'where \$([^$]+)\$ and \$([^$]+)\$ are', r'where $\1$ and $\2$ are', content)
    
    # Fix 3: Handle escaped underscores in math mode properly
    # Replace \_{ with _{  when inside math expressions
    content = re.sub(r'(\\begin{equation}.*?)\\\_', r'\1_', content, flags=re.DOTALL)
    content = re.sub(r'(\$.*?)\\\_', r'\1_', content, flags=re.DOTALL)
    
    # Fix 4: Fix specific problematic math expressions
    # Fix sigma expressions
    content = re.sub(r'\\sigma\\_{price}\^2', r'\\sigma_{\\text{price}}^2', content)
    content = re.sub(r'\\sigma\\_{vol}\^2', r'\\sigma_{\\text{vol}}^2', content)
    
    # Fix tilde expressions  
    content = re.sub(r'\\tilde{L}\\_{([^}]+)}', r'\\tilde{L}_{\\text{\1}}', content)
    
    # Fix sum expressions
    content = re.sub(r'\\sum\\_{([^}]+)}', r'\\sum_{\1}', content)
    
    # Fix hat expressions
    content = re.sub(r'\\hat{y}\\_{([^}]+)}', r'\\hat{y}_{\1}', content)
    content = re.sub(r'\\hat{P}\\_{([^}]+)}', r'\\hat{P}_{\1}', content)
    
    # Fix mathbf expressions
    content = re.sub(r'\\mathbf{([^}]+)}\\_{([^}]+)}', r'\\mathbf{\1}_{\2}', content)
    content = re.sub(r'\\mathbf{h}\\_{([^}]+)}', r'\\mathbf{h}_{\1}', content)
    content = re.sub(r'\\mathbf{x}\\_{([^}]+)}', r'\\mathbf{x}_{\1}', content)
    
    # Fix 5: Remove any remaining escaped underscores in math mode
    # This is a more general fix for any remaining \_{ patterns in equations
    def fix_equation_underscores(match):
        equation_content = match.group(1)
        # Replace \_ with _ inside equations
        equation_content = equation_content.replace('\\_', '_')
        return f'\\begin{{equation}}\n{equation_content}\n\\end{{equation}}'
    
    content = re.sub(r'\\begin{equation}\n(.*?)\n\\end{equation}', fix_equation_underscores, content, flags=re.DOTALL)
    
    # Fix 6: Handle inline math expressions
    def fix_inline_math(match):
        math_content = match.group(1)
        math_content = math_content.replace('\\_', '_')
        return f'${math_content}$'
    
    content = re.sub(r'\$([^$]+)\$', fix_inline_math, content)
    
    # Fix 7: Fix specific problem areas
    # Replace common problematic patterns
    content = re.sub(r'y_t\^\{([^}]+)\}', r'y_t^{\\text{\1}}', content)
    content = re.sub(r'r_t = r', r'r_t = r', content)  # This one is fine
    
    # Fix 8: Remove any stray backticks
    content = content.replace('```', '')
    
    # Fix 9: Specific fixes for known problem lines
    content = re.sub(r'where \$\\sigma_\{price\}\^2\$ and \$\\sigma_\{vol\}\^2\$ are', 
                    r'where $\\sigma_{\\text{price}}^2$ and $\\sigma_{\\text{vol}}^2$ are', content)
    
    # Fix 10: Clean up any double-escaped characters
    content = re.sub(r'\\\\\_', r'_', content)
    
    # Fix 11: Handle text mode subscripts that should be in math mode
    content = re.sub(r'\\text\{([^}]*?)\\\_([^}]*?)\}', r'\\text{\1_\2}', content)
    
    # Fix 12: Fix incomplete equation environments
    # Add missing \end{equation} tags where equations are not properly closed
    lines = content.split('\n')
    fixed_lines = []
    in_equation = False
    
    for line in lines:
        if '\\begin{equation}' in line:
            in_equation = True
        elif '\\end{equation}' in line:
            in_equation = False
        elif in_equation and line.strip() == '':
            # Empty line in equation, likely needs closing
            fixed_lines.append('\\end{equation}')
            fixed_lines.append('')
            in_equation = False
            continue
        elif in_equation and line.strip().startswith('where ') and '$' in line:
            # This should be outside the equation
            fixed_lines.append('\\end{equation}')
            fixed_lines.append('')
            in_equation = False
    
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix 13: Clean up Mermaid diagram code that shouldn't be there
    content = re.sub(r'```mermaid.*?```', '', content, flags=re.DOTALL)
    
    # Fix 14: Remove any remaining triple backticks
    content = re.sub(r'```[a-z]*\n(.*?)\n```', r'\1', content, flags=re.DOTALL)
    
    return content

def main():
    """Apply mathematical expression fixes."""
    print("Fixing mathematical expressions in comprehensive LaTeX...")
    
    try:
        fixed_content = fix_math_expressions()
        
        # Write the fixed version
        base_dir = Path(__file__).resolve().parents[1]
        with open(base_dir / 'paper_comprehensive_fixed.tex', 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Created fixed version: paper_comprehensive_fixed.tex")
        print(f"File size: {len(fixed_content)} characters")
        
    except Exception as e:
        print(f"Error fixing expressions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
