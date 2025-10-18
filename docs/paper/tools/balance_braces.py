#!/usr/bin/env python3
"""
Fix brace imbalance and create final Overleaf-ready version.
"""
import re
from pathlib import Path

def create_final_overleaf_version():
    """Create the final, cleanest possible version for Overleaf."""
    
    base_dir = Path(__file__).resolve().parents[1]  # docs/paper
    with open(base_dir / 'paper_clean_v2.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Start fresh with a more conservative approach
    # Only make essential changes for Overleaf compatibility
    
    # 1. Replace Unicode characters systematically
    unicode_map = {
        'σ': '\\sigma', 'π': '\\pi', 'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma',
        'δ': '\\delta', 'ε': '\\epsilon', 'θ': '\\theta', 'λ': '\\lambda', 'μ': '\\mu',
        'ρ': '\\rho', 'τ': '\\tau', 'φ': '\\phi', 'ω': '\\omega', '…': '\\ldots'
    }
    
    for unicode_char, latex_cmd in unicode_map.items():
        content = content.replace(unicode_char, latex_cmd)
    
    # 2. Fix obvious math mode issues
    content = re.sub(r'\\\\_\{', r'_{', content)  # Fix escaped underscores
    content = re.sub(r'\\\\_', r'_', content)     # Fix escaped underscores
    
    # 3. Remove problematic elements
    content = re.sub(r'```[a-zA-Z]*', '', content)  # Remove code block markers
    content = re.sub(r'style[^\\n]*', '', content)  # Remove CSS-like content
    content = re.sub(r'class[^\\n]*', '', content)  # Remove class definitions
    
    # 4. Clean up excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 5. Fix display math
    content = re.sub(r'\$\$(.*?)\$\$', r'\\[\n\\1\n\\]', content, flags=re.DOTALL)
    
    return content

def main():
    """Create final Overleaf version."""
    print("Creating final Overleaf-compatible version...")
    
    try:
        final_content = create_final_overleaf_version()
        
        base_dir = Path(__file__).resolve().parents[1]
        with open(base_dir / 'paper_overleaf_final.tex', 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"Created final Overleaf version: paper_overleaf_final.tex")
        print(f"Final version size: {len(final_content)} characters")
        
        # Check brace balance in final version
        final_open = final_content.count('{')
        final_close = final_content.count('}')
        print(f"Final brace count: {final_open} open, {final_close} close")
        
    except Exception as e:
        print(f"Error creating final version: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
