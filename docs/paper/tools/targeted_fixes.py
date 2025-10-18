#!/usr/bin/env python3
"""
Apply only targeted fixes for specific LaTeX issues without breaking the document structure.
"""
import re
from pathlib import Path

def apply_targeted_fixes():
    """Apply minimal targeted fixes only."""
    
    base_dir = Path(__file__).resolve().parents[1]  # docs/paper
    with open(base_dir / 'paper_final.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Only replace specific Unicode characters that cause errors
    unicode_fixes = {
        'σ': '\\sigma',
        'π': '\\pi', 
        'α': '\\alpha',
        'β': '\\beta',
        'μ': '\\mu',
        'θ': '\\theta',
        'λ': '\\lambda',
        'ρ': '\\rho',
        'γ': '\\gamma',
        'δ': '\\delta',
        'ε': '\\epsilon',
        'τ': '\\tau',
        'φ': '\\phi',
        'ω': '\\omega',
        '…': '\\ldots'
    }
    
    for unicode_char, latex_command in unicode_fixes.items():
        content = content.replace(unicode_char, latex_command)
    
    # Fix 2: Fix specific bad equation delimiters that we know are problematic
    # Remove any stray \eqno commands
    content = re.sub(r'\\eqno\s*', '', content)
    
    # Fix 3: Clean up any remaining triple backticks or code block remnants
    content = re.sub(r'```[a-zA-Z]*\n?', '', content)
    content = re.sub(r'\n```\n?', '\n', content)
    
    # Fix 4: Fix specific problematic math expressions mentioned in the log
    # Handle cases where display math markers are incomplete
    content = re.sub(r'\$\$([^$]+)\$\$', r'\\[\n\\1\n\\]', content)
    
    # Fix 5: Clean up excessive whitespace that might cause issues
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    # Fix 6: Ensure proper spacing around math environments
    content = re.sub(r'([^\n])\\begin{equation}', r'\\1\n\n\\begin{equation}', content)
    content = re.sub(r'\\end{equation}([^\n])', r'\\end{equation}\n\n\\1', content)
    
    return content

def main():
    """Apply targeted fixes."""
    print("Applying targeted LaTeX fixes...")
    
    try:
        fixed_content = apply_targeted_fixes()
        
        # Write the targeted fix version
        base_dir = Path(__file__).resolve().parents[1]
        with open(base_dir / 'paper_clean_v2.tex', 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Created targeted fix version: paper_clean_v2.tex")
        print(f"File size: {len(fixed_content)} characters")
        
    except Exception as e:
        print(f"Error applying targeted fixes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
