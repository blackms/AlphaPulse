# Paper: struttura e istruzioni

Questa cartella contiene tutte le risorse del paper (sorgenti LaTeX/PDF/MD), gli script di supporto e i log di compilazione.

## Struttura
- `docs/paper/`
  - Sorgenti/artefatti: `paper*.tex`, `paper*.pdf`, `paper*.aux|log|out`, `paper.md`
  - `tools/`: script per conversione/cleanup (Python)
  - `logs/`: log delle varie compilazioni/verifiche

## Compilazione rapida (LaTeX)
Puoi compilare direttamente uno dei sorgenti principali, ad esempio `paper_overleaf_final.tex` oppure `paper_final.tex`.

- Con `latexmk` (consigliato):
  - Dalla root del repo: `latexmk -pdf docs/paper/paper_overleaf_final.tex`
  - Oppure dentro la cartella: `cd docs/paper && latexmk -pdf paper_overleaf_final.tex`

- Con `pdflatex` (fallback):
  - `cd docs/paper`
  - `pdflatex -interaction=nonstopmode paper_overleaf_final.tex`
  - Esegui `pdflatex` una seconda volta per allineare riferimenti

File di log e artefatti temporanei verranno creati in questa stessa cartella; ulteriori log “diagnostici” sono in `docs/paper/logs/`.

## Conversione da Markdown a LaTeX
Script disponibili (in `docs/paper/tools/`) per generare LaTeX da `paper.md`:
- `convert_md_to_latex.py`
- `create_comprehensive_latex.py`

I path sono già relativi a `docs/paper/` e gli output finiscono in questa cartella.

Esempio di esecuzione (dalla root):
- `python docs/paper/tools/convert_md_to_latex.py`
- `python docs/paper/tools/create_comprehensive_latex.py`

## Pulizia artefatti
- Con `latexmk`: `latexmk -C` (dentro `docs/paper`) per rimuovere artefatti di compilazione
- Manuale: elimina file `*.aux`, `*.log`, `*.out` non più necessari

## Log
- Log di esecuzioni/diagnostica sono in `docs/paper/logs/` (es. `clean_compilation.log`, `overleaf_test.log`, etc.)
- In caso di errori LaTeX, controlla il relativo `*.log` accanto al `.tex` compilato e/o i file in `logs/`

## Note
- Gli script in `tools/` sono utility ad-hoc per questo paper e potrebbero richiedere Python 3.11+ (coerente col progetto principale).
- Se desideri, possiamo aggiornare noi i path hard-coded negli script per usare percorsi relativi a `docs/paper/`.
