# AI Labour Market Platform

> Shifting the question from *"Is this candidate good for this job?"*  
> to *"Where does this candidate create the highest economic value locally?"*

3rd place — EUonAIR Urban Innovation Challenge Hackathon, Lyon 2026  
Team: Bora Çakır · Cyriane Rouillon · Louis Millan · Aditya Singh

---

## What it does

A bilingual (EN/FR) Streamlit app with three modules:

**Recommendations** — Upload a CV (PDF or DOCX). The app extracts skills via OpenAI embeddings and matches them against French BMO regional labour demand data across 96 departments, surfacing the roles where the candidate creates the most economic value locally.

**Job Match** — Semantic CV ↔ job description scoring using cosine similarity on OpenAI embeddings. Goes beyond keyword matching to surface implicit competencies.

**Predictive Modeling** — Upload BMO time-series data (2020–2025) to forecast regional job demand by sector. Linear regression and Random Forest models with YoY trend analysis and configurable forecast horizon.

---

## Data

Built on French BMO (Besoin en Main-d'Œuvre) data — an annual survey by Pôle Emploi covering workforce demand across all 96 French departments and ~200 job families.

- `gap_metier.csv` — demand by job title and department
- `gap_sector.csv` — demand aggregated by sector

---

## Setup
```bash
git clone https://github.com/TheEmeraldAgent/ai-labour-market-platform
cd ai-labour-market-platform
pip install -r requirements.txt
```

Create a `.streamlit/secrets.toml` file:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"
```

Run:
```bash
streamlit run app.py
```

---

## Stack

`Python` `Streamlit` `OpenAI API` `Scikit-learn` `Pandas` `NumPy` `RapidFuzz` `PyPDF` `Matplotlib`

---

## Context

Built in 72 hours at the EUonAIR Urban Innovation Challenge (Lyon, Feb 2026), a pan-European hackathon focused on AI solutions for vulnerable job seekers.
