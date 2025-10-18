#!/usr/bin/env python3
"""
Create comprehensive LaTeX version from the complete markdown paper.
"""
import re
from pathlib import Path

def convert_comprehensive_md_to_latex():
    """Convert the full markdown paper to LaTeX."""
    
    # Read the complete markdown file
    base_dir = Path(__file__).resolve().parents[1]  # docs/paper
    with open(base_dir / 'paper.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # LaTeX header
    latex_content = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{geometry}
\usepackage{natbib}
\usepackage{url}
\usepackage{hyperref}
\usepackage{longtable}

% Page layout
\geometry{margin=1in}

\title{Streamlined Hierarchical Reinforcement Learning for Algorithmic Trading: Architecture Simplification and Empirical Validation}

\author{Alessio Rocchi\\
AIGen Consult\\
\texttt{alessio@aigenconsult.com}}

\date{\today}

\begin{document}

\maketitle

"""

    # Process content
    content = md_content
    
    # Remove title from content (already in header)
    content = re.sub(r'^# .+$', '', content, flags=re.MULTILINE)
    
    # Remove author/email metadata
    content = re.sub(r'\*\*Author:\*\* .+', '', content)
    content = re.sub(r'\*\*Email:\*\* .+', '', content)
    
    # Add abstract
    abstract_match = re.search(r'## 1\. Abstract\n\n(.*?)\n\n##', content, re.DOTALL)
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        latex_content += f"\\begin{{abstract}}\n{abstract_text}\n\\end{{abstract}}\n\n"
        # Remove abstract from content
        content = content.replace(abstract_match.group(0), '\n\n##')
    
    # Add keywords
    keywords_match = re.search(r'\*\*Keywords:\*\* (.+)', content)
    if keywords_match:
        keywords = keywords_match.group(1)
        latex_content += f"\\textbf{{Keywords:}} {keywords}\n\n"
        content = re.sub(r'\*\*Keywords:\*\* .+', '', content)
    
    # Convert headers
    content = re.sub(r'^## (\d+)\. (.+)$', r'\\section{\2}', content, flags=re.MULTILINE)
    content = re.sub(r'^### (\d+\.\d+) (.+)$', r'\\subsection{\2}', content, flags=re.MULTILINE)
    content = re.sub(r'^### (.+)$', r'\\subsection{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^#### (.+)$', r'\\subsubsection{\1}', content, flags=re.MULTILINE)
    
    # Convert math blocks - handle both ```math and $$ formats
    content = re.sub(r'```math\n(.*?)\n```', r'\\begin{equation}\n\1\n\\end{equation}', content, flags=re.DOTALL)
    content = re.sub(r'\$\$(.*?)\$\$', r'\\begin{equation}\n\1\n\\end{equation}', content, flags=re.DOTALL)
    
    # Convert bold text
    content = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', content)
    
    # Convert italic text
    content = re.sub(r'\*([^*\n]+)\*', r'\\textit{\1}', content)
    
    # Convert bullet lists to itemize
    lines = content.split('\n')
    result_lines = []
    in_itemize = False
    in_enumerate = False
    
    for line in lines:
        # Handle bullet points
        if re.match(r'^[\*\-] ', line):
            if not in_itemize and not in_enumerate:
                result_lines.append('\\begin{itemize}')
                in_itemize = True
            elif in_enumerate:
                result_lines.append('\\end{enumerate}')
                result_lines.append('\\begin{itemize}')
                in_enumerate = False
                in_itemize = True
            result_lines.append(re.sub(r'^[\*\-] (.+)$', r'\\item \1', line))
            
        # Handle numbered lists
        elif re.match(r'^\d+\. ', line):
            if not in_enumerate and not in_itemize:
                result_lines.append('\\begin{enumerate}')
                in_enumerate = True
            elif in_itemize:
                result_lines.append('\\end{itemize}')
                result_lines.append('\\begin{enumerate}')
                in_itemize = False
                in_enumerate = True
            result_lines.append(re.sub(r'^\d+\. (.+)$', r'\\item \1', line))
            
        else:
            # End lists when we encounter non-list content
            if (in_itemize or in_enumerate) and line.strip() != '':
                if in_itemize:
                    result_lines.append('\\end{itemize}')
                    in_itemize = False
                if in_enumerate:
                    result_lines.append('\\end{enumerate}')
                    in_enumerate = False
            result_lines.append(line)
    
    # Close any remaining lists
    if in_itemize:
        result_lines.append('\\end{itemize}')
    if in_enumerate:
        result_lines.append('\\end{enumerate}')
        
    content = '\n'.join(result_lines)
    
    # Convert tables to longtable format
    def convert_table(match):
        table_content = match.group(0)
        lines = table_content.strip().split('\n')
        if len(lines) < 3:
            return table_content
            
        header = lines[0]
        separator = lines[1] 
        rows = lines[2:]
        
        # Extract headers
        header_cells = [cell.strip() for cell in header.split('|') if cell.strip()]
        if not header_cells:
            return table_content
            
        cols = len(header_cells)
        col_spec = 'l' * cols
        
        latex_table = f"\\begin{{longtable}}{{{col_spec}}}\\n\\toprule\\n"
        latex_table += ' & '.join([f"\\textbf{{{cell}}}" for cell in header_cells]) + ' \\\\\\n\\midrule\\n'
        
        # Process data rows
        for row in rows:
            row_cells = [cell.strip() for cell in row.split('|') if cell.strip()]
            if row_cells and len(row_cells) == cols:
                latex_table += ' & '.join(row_cells) + ' \\\\\\n'
        
        latex_table += "\\bottomrule\\n\\end{longtable}\\n"
        return latex_table
    
    # Find and convert tables
    content = re.sub(r'\|[^\n]+\|.*?(?=\n\n|$)', convert_table, content, flags=re.DOTALL | re.MULTILINE)
    
    # Handle special LaTeX characters
    content = content.replace('&', '\\&')
    content = content.replace('%', '\\%')
    content = content.replace('#', '\\#')
    content = content.replace('_', '\\_')
    
    # Fix common issues with underscores in math mode
    content = re.sub(r'\\textbf\{([^}]*)\\_([^}]*)\}', r'\\textbf{\1_\2}', content)
    
    # Add content and close document
    latex_content += content
    latex_content += "\n\n\\section{References}\n\n% Bibliography would be added here\n\n\\end{document}"
    
    return latex_content

def main():
    """Generate comprehensive LaTeX version."""
    print("Creating comprehensive LaTeX version...")
    
    try:
        latex_content = convert_comprehensive_md_to_latex()
        
        base_dir = Path(__file__).resolve().parents[1]  # docs/paper
        with open(base_dir / 'paper_comprehensive.tex', 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"Created comprehensive LaTeX file: {len(latex_content)} characters")
        print(f"File: {base_dir / 'paper_comprehensive.tex'}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
