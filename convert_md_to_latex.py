#!/usr/bin/env python3
"""
Convert AlphaPulse paper from Markdown to LaTeX format.
"""

import re
import sys
from pathlib import Path

def convert_markdown_to_latex(md_content):
    """Convert markdown content to LaTeX format."""
    
    # Start with basic LaTeX structure
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
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{longtable}

% Page layout
\geometry{margin=1in}

% Custom commands
\newcommand{\mathbf{w}}{\mathbf{w}}
\newcommand{\mathbf{x}}{\mathbf{x}}
\newcommand{\mathbf{h}}{\mathbf{h}}

\begin{document}

"""
    
    # Extract title
    title_match = re.search(r'^# (.+)$', md_content, re.MULTILINE)
    if title_match:
        title = title_match.group(1)
        latex_content += f"\\title{{{title}}}\n\n"
    
    # Extract author and email
    author_match = re.search(r'\*\*Author:\*\* (.+)', md_content)
    email_match = re.search(r'\*\*Email:\*\* (.+)', md_content)
    if author_match:
        author = author_match.group(1)
        if email_match:
            email = email_match.group(1)
            latex_content += f"\\author{{{author}\\\\AIGen Consult\\\\\\texttt{{{email}}}}}\n\n"
        else:
            latex_content += f"\\author{{{author}}}\n\n"
    
    latex_content += "\\date{\\today}\n\n"
    latex_content += "\\maketitle\n\n"
    
    # Process content sections
    content = md_content
    
    # Remove title, author, email, keywords from content
    content = re.sub(r'^# .+$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\*\*Author:\*\* .+', '', content)
    content = re.sub(r'\*\*Email:\*\* .+', '', content)
    content = re.sub(r'\*\*Keywords:\*\* .+', '', content)
    
    # Convert headers
    content = re.sub(r'^## (\d+)\. (.+)$', r'\\section{\2}', content, flags=re.MULTILINE)
    content = re.sub(r'^### (\d+\.\d+) (.+)$', r'\\subsection{\2}', content, flags=re.MULTILINE)
    content = re.sub(r'^### (.+)$', r'\\subsection{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^#### (.+)$', r'\\subsubsection{\1}', content, flags=re.MULTILINE)
    
    # Convert math blocks
    content = re.sub(r'```math\n(.*?)\n```', r'\\begin{equation}\n\1\n\\end{equation}', content, flags=re.DOTALL)
    content = re.sub(r'\$\$(.*?)\$\$', r'\\begin{equation}\n\1\n\\end{equation}', content, flags=re.DOTALL)
    
    # Convert inline math (keep as is)
    # content = re.sub(r'\$([^$]+)\$', r'$\1$', content)
    
    # Convert bold text
    content = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', content)
    
    # Convert italic text  
    content = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', content)
    
    # Convert bullet lists
    content = re.sub(r'^[\*\-] (.+)$', r'\\item \1', content, flags=re.MULTILINE)
    
    # Wrap consecutive items in itemize
    lines = content.split('\n')
    result_lines = []
    in_itemize = False
    
    for line in lines:
        if line.strip().startswith('\\item'):
            if not in_itemize:
                result_lines.append('\\begin{itemize}')
                in_itemize = True
            result_lines.append(line)
        else:
            if in_itemize:
                result_lines.append('\\end{itemize}')
                in_itemize = False
            result_lines.append(line)
    
    if in_itemize:
        result_lines.append('\\end{itemize}')
    
    content = '\n'.join(result_lines)
    
    # Convert numbered lists
    content = re.sub(r'^(\d+)\. (.+)$', r'\\item \2', content, flags=re.MULTILINE)
    
    # Convert tables (basic conversion)
    def convert_table(match):
        table_content = match.group(0)
        lines = table_content.strip().split('\n')
        if len(lines) < 3:
            return table_content
            
        header = lines[0]
        separator = lines[1]
        rows = lines[2:]
        
        # Count columns
        cols = len([col.strip() for col in header.split('|') if col.strip()])
        col_spec = 'l' * cols
        
        latex_table = f"\\begin{{longtable}}{{{col_spec}}}\n\\toprule\n"
        
        # Header
        header_cells = [cell.strip() for cell in header.split('|') if cell.strip()]
        latex_table += ' & '.join(header_cells) + ' \\\\\n\\midrule\n'
        
        # Rows
        for row in rows:
            row_cells = [cell.strip() for cell in row.split('|') if cell.strip()]
            if row_cells:
                latex_table += ' & '.join(row_cells) + ' \\\\\n'
        
        latex_table += "\\bottomrule\n\\end{longtable}\n"
        return latex_table
    
    # Find and convert tables
    content = re.sub(r'\|[^|]+\|.*?(?=\n\n|\Z)', convert_table, content, flags=re.DOTALL | re.MULTILINE)
    
    # Handle special characters
    content = content.replace('#', '\\#')
    content = content.replace('&', '\\&')
    content = content.replace('%', '\\%')
    content = content.replace('_', '\\_')
    
    # Add abstract if found
    abstract_match = re.search(r'## 1\. Abstract\n\n(.*?)\n\n##', content, re.DOTALL)
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        abstract_latex = f"\\begin{{abstract}}\n{abstract_text}\n\\end{{abstract}}\n\n"
        content = content.replace(abstract_match.group(0), f"\\section{{Introduction}}\n\n##")
        latex_content += abstract_latex
    
    latex_content += content
    latex_content += "\n\n\\end{document}"
    
    return latex_content

def main():
    """Main conversion function."""
    md_file = Path('/Users/a.rocchi/Projects/Personal/AlphaPulse/paper.md')
    latex_file = Path('/Users/a.rocchi/Projects/Personal/AlphaPulse/paper.tex')
    
    print(f"Converting {md_file} to {latex_file}...")
    
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        latex_content = convert_markdown_to_latex(md_content)
        
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"Successfully converted to {latex_file}")
        print(f"LaTeX file size: {len(latex_content)} characters")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()