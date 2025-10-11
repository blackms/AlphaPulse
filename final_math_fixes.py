#!/usr/bin/env python3
"""
Final fixes for remaining mathematical expressions in LaTeX.
"""
import re

def apply_final_fixes():
    """Apply final targeted fixes for mathematical expressions."""
    
    with open('/Users/a.rocchi/Projects/Personal/AlphaPulse/paper_comprehensive_fixed.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Replace orphaned 'math' with proper equation environment
    content = re.sub(r'\s+math\s*\n\s*([^\\]*\\tilde{L}[^\n]+)\s*\n', r'\n\\begin{equation}\n\1\n\\end{equation}\n', content)
    
    # Fix 2: Find equations that are not properly wrapped
    # Pattern: standalone math expressions that should be in equation environment
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check if line contains math but is not in equation environment
        if (stripped.startswith('\\tilde{') or stripped.startswith('\\sum_') or 
            stripped.startswith('\\pi_') or stripped.startswith('R_')) and not stripped.startswith('\\begin{equation}'):
            
            # Check if previous and next lines suggest this should be an equation
            prev_line = lines[i-1].strip() if i > 0 else ''
            next_line = lines[i+1].strip() if i < len(lines)-1 else ''
            
            if prev_line == '' and next_line == '':
                # This looks like a standalone equation
                fixed_lines.append('\\begin{equation}')
                fixed_lines.append(line)
                fixed_lines.append('\\end{equation}')
                continue
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix 3: Handle specific problematic patterns
    # Fix regime set notation  
    content = re.sub(r'where \$R = \\{.*?\\}\$ represents', 
                    r'where $R = \\{\\text{bull-low-vol}, \\text{bull-high-vol}, \\text{bear-low-vol}, \\text{bear-high-vol}, \\text{sideways-low-vol}, \\text{sideways-high-vol}\\}$ represents',
                    content)
    
    # Fix 4: Clean up any remaining mermaid or code block remnants
    content = re.sub(r'graph TD.*?class EOA hrl_execution;', '', content, flags=re.DOTALL)
    
    # Fix 5: Fix specific math expressions that are causing issues
    content = re.sub(r'\\mathbf\{([^}]+)\}([^\\]+)', r'\\mathbf{\1}\2', content)
    
    # Fix 6: Ensure proper math mode for mathematical operators
    content = re.sub(r'([^\\])(operatorname\{[^}]+\})', r'\1\\operatorname{\2}', content)
    
    return content

def main():
    """Apply final mathematical fixes."""
    print("Applying final mathematical expression fixes...")
    
    try:
        fixed_content = apply_final_fixes()
        
        # Write the final fixed version
        with open('/Users/a.rocchi/Projects/Personal/AlphaPulse/paper_final.tex', 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Created final version: paper_final.tex")
        print(f"File size: {len(fixed_content)} characters")
        
    except Exception as e:
        print(f"Error applying final fixes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()