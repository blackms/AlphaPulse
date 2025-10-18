# Perizia Tecnica di Stima – AlphaPulse

- Incarico: perizia tecnica indipendente del software AlphaPulse per finalità di capitalizzazione dell’IP.
- Ruolo: CTU/Perito indipendente.
- Data di valutazione (freeze): 2025-09-12.
- Repo ref: commit HEAD `0e3ce1c`.
- Titolare dell’IP: aigensolutions srl.
- Sviluppatore principale/contributor: Alessio Rocchi (socio della società).

Nota d’indipendenza: la stima è condotta su base documentale e tecnica; l’assenza di alcune evidenze (finanziarie/legali) comporta maggiore incertezza. La stima finale richiede integrazioni elencate in coda.

## Oggetto e Perimetro
- Ambito software: backend Python (FastAPI), moduli `api/`, `backtesting/`, `ml/`, `services/`, migrazioni Alembic e CI.
- Infrastruttura/stack: Python ≥3.11, FastAPI, PostgreSQL, Redis. Frontend Next.js (menzionato in contesto; non presente in questo repo).
- Ambiente/operatività: repo con CI GitHub Actions; scripts/docker presenti.

## Sintesi Esecutiva (provvisoria)
- Valore puntuale e range (Metodo del Costo – prima stima): punto ≈ 198.000 €, range ≈ 92.000–333.000 €.
- Indicazione potenziale triangolata (Costo + Reddito + Mercato): punto ≈ 580.000 €, range ≈ 300.000–1.000.000 €.
- Metodologie: Costo (RCN), Reddito (Relief-from-Royalty e DCF attribuito all’IP), Mercato (comparabili). Riconciliazione pesata e sensitività su parametri chiave.
- Presidi richiesti: coverage e qualità codice, scansioni sicurezza, SBOM/licenze, titolarità IP, forecast e parametri finanziari (WACC, tax, crescita).
- Rischi residui principali: copertura test bassa (report attuale ≈7,3%), assenza evidenze SAST/SBOM, parametri finanziari non forniti, documentazione titolarità da acquisire.

## Due Diligence
### Tecnica
- Test: 84 file di test individuati (`src/alpha_pulse/tests`).
- Coverage da `coverage.xml`: lines-covered 861 / lines-valid 11.850 → line-rate ≈ 7,27%.
- Tooling configurato: `black`, `flake8`, `mypy`; non risultano `bandit`/`safety` in `pyproject.toml`.
- CI: workflow presenti (`.github/workflows/python-app.yml`, `pages.yml`).
- Migrazioni: presenti (`migrations/`, `alembic.ini`).
- Architettura: app FastAPI (`src/alpha_pulse/api/main.py`), routers e websockets; moduli backtesting e ML.
- Osservazioni: il delta fra numero test e coverage suggerisce incongruenza di configurazione/target o report obsoleto; raccomandata verifica esecuzione e inclusione path corretti in `--cov`.
 - Timeline repo: 2025-01-03 → 2025-09-08 (≈ 8 mesi); commit totali ≈ 846.

### Legale/Ownership
- Dichiarazione fornita: l’IP è di proprietà di aigensolutions srl; sviluppato da Alessio Rocchi (socio).
- Documenti richiesti: chain of title/assignments, accordi con i contributor, policy OSS e conformità licenze.

### Commerciale
- Dati non presenti nel repo. Richiesti: modello di business, pricing, ricavi storici/forecast, concentrazione clienti, churn/retention, pipeline/contratti.

## Metodologie e Calcoli (schemi)
### 1) Costo – Replacement Cost New (RCN)
Formule:
- RCN = Σ(FTE_mesi_i × costo_mensile_i) + tooling + dati + overhead(15%)
- Obsolescenza_tot = funzionale% + tecnologica% + economica% (cap 100%)
- Valore_Costo = RCN × (1 − Obsolescenza_tot)

Assunzioni e basi di calcolo (repo):
- LOC non‑test (`src/alpha_pulse`, wc -l): ≈ 141.983 linee.
- Produttività 1 dev con IA: 4.000–8.000 LOC/mese ⇒ mesi equivalenti ≈ 35,5 (basso prod.) – 17,7 (alto prod.).
- Costo mensile ingegnere: 6.500–9.500 €; overhead 15% su (ingegneria + tooling + dati).
- Tooling/servizi: 8.000–15.000 €; Dati/labeling: 0–10.000 €.
- Obsolescenza: 35% (scenario LOW), 25% (BASE), 20% (HIGH).

Stima scenari:

| Scenario | Mesi eq. | Costo mensile | Tooling | Dati | Overhead | RCN lordo | Obsolescenza | Valore_Costo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LOW | 17,7 | 6.500 € | 8.000 € | 0 € | 15% | 141.865 € | 35% | 92.212 € |
| BASE | 26,6 | 8.000 € | 11.500 € | 5.000 € | 15% | 263.896 € | 25% | 197.922 € |
| HIGH | 35,5 | 9.500 € | 15.000 € | 10.000 € | 15% | 416.541 € | 20% | 333.233 € |

Note:
- La stima usa LOC totali (non normalizzati per commenti/blank); la produttività con IA è tarata per compensare l’effetto.
- La perizia finale può rifinire i parametri (costi, tooling, obsolescenza) con evidenze documentali.

### 2) Reddito – Relief-from-Royalty (RfR)
Formule:
- Royalty_t = Ricavi_t × tasso_royalty
- Risparmio_post_tax_t = Royalty_t × (1 − tax)
- TV (Gordon) = Risparmio_{n+1} ÷ (WACC − g)
- Valore_RfR = Σ Risparmio_post_tax_t / (1+WACC)^t + TV/(1+WACC)^n

Schema (da compilare):
| Anno | Ricavi | Tasso royalty | Royalty | Tax | Risparmio post-tax | Fatt. att. | Valore attuale |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | … | … | … | … | … | … | … |
| 2 | … | … | … | … | … | … | … |
| … | … | … | … | … | … | … | … |
| n | … | … | … | … | … | … | … |
| TV | — | — | — | — | g=… |  | … |
| Totale | | | | | | | Somma |

Parametri richiesti: forecast ricavi, tasso royalty di mercato per categoria, tax, WACC, crescita g.

Assunzioni (base, indicative) e risultato:
- Mix: B2C 2.000 utenti × 39 €/mese; B2B 30 clienti × 12.000 €/anno ⇒ Ricavi ≈ 1,416 M€/a.
- Royalty 5%; tax 24%; WACC 22%; g 2% (perpetuity con crescita modesta).
- Risultato base: Valore_RfR ≈ 0,32 M€; range indicativo: 0,26–0,38 M€.

### 3) Reddito – DCF attribuito all’IP
Assunzioni: fattore di attribuzione all’IP α ∈ [0,1].
Formule:
- FCF_IP_t = FCF_totale_t × α
- TV (Gordon) = FCF_IP_{n+1} ÷ (WACC − g)
- Valore_DCF = Σ FCF_IP_t / (1+WACC)^t + TV/(1+WACC)^n

Schema (da compilare):
| Anno | FCF totale | α | FCF attrib. IP | Fatt. att. | Valore attuale |
|---:|---:|---:|---:|---:|---:|
| 1 | … | … | … | … | … |
| 2 | … | … | … | … | … |
| … | … | … | … | … | … |
| n | … | … | … | … | … |
| TV | — | — | g=… |  | … |
| Totale | | | | | Somma |

Parametri richiesti: FCF forecast, α, WACC, g.

### 4) Approccio di Mercato – Comparabili
- Fonti: deal di licensing, multipli M&A, comparabili pubblici (stesso dominio/scala).
- Normalizzazione: per dimensione, crescita, margini, perimetro IP.
- Output: range di multipli/royalty e valore indicativo.

Stima indicativa (multipli su ARR attribuibile all'IP):
- ARR Y3 ipotizzata ≈ 1,4 M€; fattore attribuzione IP α ≈ 0,6 ⇒ 0,84 M€.
- Multipli IP-adjusted 2–4× ⇒ 1,7–3,4 M€ lordo “going concern”; sconto IP-only 50–70% ⇒ 0,5–1,7 M€.

### 5) Riconciliazione e Sensitività
Riconciliazione (provvisoria):
| Metodo | Valore (M€) | Peso | Contributo (M€) |
|---|---:|---:|---:|
| Costo (RCN) | 0,198 | 30% | 0,059 |
| RfR (base) | 0,320 | 40% | 0,128 |
| Mercato (base) | 1,200 | 30% | 0,360 |
| Totale | | | 0,547 |

Indicazione: punto provvisorio ~0,55–0,60 M€ (arrotondato 0,58 M€), range 0,30–1,00 M€.

Sensitività RfR (variazioni su parametri chiave):
| Parametro | -20% | -10% | Base | +10% | +20% |
|---|---:|---:|---:|---:|---:|
| Royalty (r) [lineare] | 0,256 M€ (-20%) | 0,288 M€ (-10%) | 0,320 M€ | 0,352 M€ (+10%) | 0,384 M€ (+20%) |
| WACC (w) [pp] (g=2%) | 0,400 M€ (-4pp) | 0,356 M€ (-2pp) | 0,320 M€ | 0,291 M€ (+2pp) | 0,267 M€ (+4pp) |
| Pricing (ARPU) [lineare] | 0,256 M€ (-20%) | 0,288 M€ (-10%) | 0,320 M€ | 0,352 M€ (+10%) | 0,384 M€ (+20%) |

## Conclusioni, Limiti, Eventi Successivi
- La stima finale è subordinata a evidenze tecniche/finanziarie/legali integrative elencate sotto.
- Limiti: report coverage potenzialmente non aggiornato; assenza scansioni sicurezza e SBOM; assenza documenti di titolarità; parametri finanziari non dichiarati.
- Eventi successivi: eventuali rilasci o modifiche sostanziali al codice dopo il commit di riferimento richiedono aggiornamento del perimetro/stima.

## Allegati ed Evidenze (da allegare)
- Report test/coverage aggiornati (JUnit/XML, HTML `htmlcov`).
- Log `mypy`, `flake8/pylint`, `black --check`.
- Scansioni sicurezza: `bandit`, `safety/pip-audit`, SCA CI.
- SBOM (CycloneDX) con licenze e risk posture.
- ADR/architettura, runbook operativi (se presenti), release notes.

## Richieste Integrative (Must-have)
1) Finanza: ricavi storici e forecast o FCF per linea prodotto; WACC; tax; crescita; α; tasso royalty.
2) Ingegneria: FTE-mesi per componente/periodo, rate caricati, spese tooling/dati, policy overhead; obsolescenza funzionale/tecnologica/economica.
3) Legale: chain of title, accordi contributor, attestazione compliance OSS/licenze.
4) Commerciale: pricing, moats, concentrazione clienti, churn/retention, pipeline/contratti.
5) Versioning: conferma data di valutazione e commit/tag di riferimento (attuale: `0e3ce1c`).

## Raccomandazioni Operative (Qualità/Sicurezza)
- Eseguire quality gates e allegare report:
  - `pytest --cov=src/alpha_pulse --cov-report=xml:coverage.xml`
  - `mypy src` · `flake8` · `black --check .`
  - SAST/Deps: `bandit -r src` · `safety check` o `pip-audit`.
  - SBOM (CycloneDX Poetry) e licenze.
- Allineare configurazione coverage e percorsi per riflettere correttamente i moduli `src/alpha_pulse`.

—
Documento generato come scheletro operativo secondo PERIZIA-PROTO; completare i campi "DA COMPILARE" con i dati forniti dal committente e allegare le evidenze.
