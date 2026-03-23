# app.py
# Run: python -m streamlit run app.py

import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

from pypdf import PdfReader
from docx import Document
from openai import OpenAI

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# I18N (UI TEXT)
# =========================

I18N = {
    "en": {
        "app_title": "Talent Recommendations + Job Match",
        "tab_reco": "✅ Recommendations",
        "tab_match": "🎯 Job Match",
        "tab_forecast": "📈 Predictive Modeling",
        "region": "Region",
        "department": "City / Department",
        "sector": "Sector",
        "upload_cv": "Upload CV",
        "upload_csv": "Upload CSV",
        "run_forecast": "Run Forecast",
        "training_advice": "🎓 Training Advice",
        "gen_training": "Generate training advice",
        "not_enough_years": "Not enough historical years for this selection.",
        "years_available": "Years available for this selection:",
    },
    "fr": {
        "app_title": "Recommandations de métiers + Matching CV ↔ Offre",
        "tab_reco": "✅ Recommandations",
        "tab_match": "🎯 Matching CV ↔ Offre",
        "tab_forecast": "📈 Modélisation prédictive",
        "region": "Région",
        "department": "Département / Ville",
        "sector": "Secteur",
        "upload_cv": "Importer le CV",
        "upload_csv": "Importer un CSV",
        "run_forecast": "Lancer la prévision",
        "training_advice": "🎓 Conseils de formation",
        "gen_training": "Générer des recommandations de formation",
        "not_enough_years": "Historique insuffisant pour cette sélection.",
        "years_available": "Années disponibles pour cette sélection :",
    }
}

def t(key: str, lang: str) -> str:
    # fallback: English
    return I18N.get(lang, I18N["en"]).get(key, I18N["en"].get(key, key))

# =========================
# CONFIG
# =========================

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_EMBED_MODEL = "text-embedding-3-large"


# =========================
# TEXT REPAIR (mojibake)
# =========================

def _fix_mojibake_text(s: str) -> str:
    """
    Fix strings like 'RhÃ´ne' -> 'Rhône'
    """
    if not isinstance(s, str):
        return s
    if "Ã" in s or "Â" in s or "�" in s:
        try:
            return s.encode("latin1").decode("utf-8")
        except Exception:
            return s
    return s


def _fix_mojibake_df(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].map(_fix_mojibake_text)
    return df


# =========================
# ROBUST CSV PARSER FOR gap_metier.csv
# Handles unquoted commas inside metier by using:
# region, department, (metier...possibly with commas), demand_value, demand_score
# =========================

def _parse_gap_metier_manually(path: Path) -> pd.DataFrame:
    """
    Manual parse that tolerates malformed CSV rows where 'metier' contains commas
    without quotes. Assumes schema:
      region,department,metier,demand_value,demand_score
    """
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline().strip()

        # If file has an empty first line or BOM weirdness, keep reading until header found
        while header == "":
            header = f.readline().strip()

        # Expect at least these columns
        # We will NOT trust header for parsing, but we’ll use it as a sanity check.
        # header example: region,department,metier,demand_value,demand_score

        line_no = 1
        for line in f:
            line_no += 1
            line = line.strip()
            if not line:
                continue

            # Rule:
            # - split first 2 commas from left
            # - split last 2 commas from right
            try:
                left = line.split(",", 2)
                if len(left) < 3:
                    # malformed line
                    continue
                region, department, rest = left[0], left[1], left[2]

                right = rest.rsplit(",", 2)
                if len(right) < 3:
                    # malformed line
                    continue
                metier, demand_value, demand_score = right[0], right[1], right[2]

                rows.append(
                    {
                        "region": region.strip(),
                        "department": department.strip(),
                        "metier": metier.strip().strip('"'),
                        "demand_value": demand_value.strip(),
                        "demand_score": demand_score.strip(),
                    }
                )
            except Exception:
                # Skip line if anything goes wrong
                continue

    df = pd.DataFrame(rows)
    return df


def _read_csv_generic(path: Path) -> pd.DataFrame:
    """
    Generic CSV reader for other files (e.g., gap_sector.csv).
    Still resilient to some mess via on_bad_lines.
    """
    return pd.read_csv(
        path,
        sep=",",
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines="skip",
        quotechar='"',
        escapechar="\\",
    )


@st.cache_data
def load_gap_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(__file__).resolve().parent

    metier_path = base / "gap_metier.csv"
    sector_path = base / "gap_sector.csv"

    # --- Detect delimiter from header ---
    with open(metier_path, "r", encoding="utf-8-sig", errors="replace") as f:
        header_line = f.readline()

    if ";" in header_line and "," not in header_line:
        sep = ";"
    elif "," in header_line:
        sep = ","
    else:
        sep = ";"  # fallback (French datasets usually ;)

    # --- Read files using detected delimiter ---
    gap_metier = pd.read_csv(
        metier_path,
        sep=sep,
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines="skip",
    )

    # Try same separator for sector; fallback if needed
    try:
        gap_sector = pd.read_csv(
            sector_path,
            sep=sep,
            encoding="utf-8-sig",
            engine="python",
            on_bad_lines="skip",
        )
    except Exception:
        gap_sector = pd.read_csv(
            sector_path,
            sep=";" if sep == "," else ",",
            encoding="utf-8-sig",
            engine="python",
            on_bad_lines="skip",
        )

    # --- Fix mojibake ---
    def fix_text(s):
        if isinstance(s, str) and ("Ã" in s or "Â" in s):
            try:
                return s.encode("latin1").decode("utf-8")
            except Exception:
                return s
        return s

    for col in gap_metier.select_dtypes(include="object"):
        gap_metier[col] = gap_metier[col].map(fix_text)

    for col in gap_sector.select_dtypes(include="object"):
        gap_sector[col] = gap_sector[col].map(fix_text)

    # Normalize column names
    gap_metier.columns = gap_metier.columns.str.strip().str.lower()
    gap_sector.columns = gap_sector.columns.str.strip().str.lower()

    # Clean required columns
    for col in ["region", "department", "metier"]:
        if col in gap_metier.columns:
            gap_metier[col] = gap_metier[col].astype(str).str.strip()

    if "demand_score" in gap_metier.columns:
        gap_metier["demand_score"] = pd.to_numeric(
            gap_metier["demand_score"],
            errors="coerce"
        ).fillna(0.0)

    return gap_metier, gap_sector


# =========================
# FILE PARSING
# =========================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    out = []
    for page in reader.pages:
        out.append(page.extract_text() or "")
    return "\n".join(out)


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def read_uploaded_text(uploaded_file) -> str:
    b = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(b)

    if name.endswith(".docx"):
        return extract_text_from_docx(b)

    # txt fallback
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("latin1", errors="ignore")


# =========================
# OPENAI HELPERS
# =========================

def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def extract_profile(client: OpenAI, model: str, text: str, lang: str = "en") -> dict:
    """
    Strict JSON extraction (hard skills, soft skills, metier, summary)
    with language-controlled output.

    lang: "en" or "fr"
    """
    output_lang = "French" if lang == "fr" else "English"

    schema = {
        "type": "object",
        "properties": {
            "hard_skills": {"type": "array", "items": {"type": "string"}},
            "soft_skills": {"type": "array", "items": {"type": "string"}},
            "metier": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["hard_skills", "soft_skills", "metier", "summary"],
        "additionalProperties": False,
    }

    instructions = (
        f"Write everything in {output_lang}.\n\n"
        "Extract from the text:\n"
        "- hard_skills: tools/technologies/methods/certifications/domains\n"
        "- soft_skills: behaviors/capabilities (can be explicit or implicit)\n"
        "- metier: the most likely métier/job role label (if French output, use a French label)\n"
        "- summary: 2-3 lines summarizing the profile\n\n"
        "Rules:\n"
        "- Return JSON only (no markdown, no extra text).\n"
        "- Use short, canonical skill phrases (e.g., 'Python', 'Power BI', 'GDPR', 'SQL').\n"
        "- Deduplicate skills.\n"
    )

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=text[:12000],
        text={
            "format": {
                "type": "json_schema",
                "name": "cv_job_profile",
                "schema": schema,
                "strict": True,
            }
        },
    )
    return json.loads(resp.output_text)


def get_embedding(client: OpenAI, embed_model: str, text: str) -> List[float]:
    emb = client.embeddings.create(model=embed_model, input=text)
    return emb.data[0].embedding


def cosine(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# =========================
# SKILL LIST HELPERS
# =========================

def uniq(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in items or []:
        s2 = str(s).strip()
        if not s2:
            continue
        key = s2.lower()
        if key not in seen:
            seen.add(key)
            out.append(s2)
    return out


def render_bullets(title: str, items: List[str]):
    st.markdown(f"#### {title}")
    if not items:
        st.caption("—")
        return
    for it in items:
        st.write(f"• {it}")


# =========================
# DEMAND + FUZZY MATCH FOR METIER
# =========================

def demand_lookup(gap_metier_df: pd.DataFrame, metier: str, region: str, department: str):
    sub = gap_metier_df[
        (gap_metier_df["region"] == region) &
        (gap_metier_df["department"] == department)
    ]
    if sub.empty:
        return 0.0, "none", None

    exact = sub[sub["metier"] == metier]
    if not exact.empty:
        row = exact.iloc[0]
        return float(row.get("demand_score", 0.0)), "exact", row.get("metier")

    choices = sub["metier"].tolist()
    best = process.extractOne(metier, choices, scorer=fuzz.token_sort_ratio)
    if best and best[1] >= 85:
        matched = best[0]
        row = sub[sub["metier"] == matched].iloc[0]
        return float(row.get("demand_score", 0.0)), "fuzzy", matched

    return 0.0, "none", None


# =========================
# RECOMMENDATION ENGINE (gap + qualifications) + EMBEDDING CACHE
# =========================

def _get_metier_emb_cache() -> Dict[str, List[float]]:
    if "metier_emb_cache" not in st.session_state:
        st.session_state["metier_emb_cache"] = {}
    return st.session_state["metier_emb_cache"]


def _skill_overlap_score(candidate_profile: dict, metier_label: str) -> float:
    """
    Lightweight heuristic: if the metier label shares words with candidate metier/skills,
    increase score slightly. (Keeps it deterministic and cheap.)
    """
    cand_words = set()
    for x in (candidate_profile.get("metier", "") + " " + " ".join(candidate_profile.get("hard_skills", [])[:60])).lower().split():
        if len(x) >= 4:
            cand_words.add(x)

    met_words = set(w for w in str(metier_label).lower().split() if len(w) >= 4)
    if not met_words:
        return 0.0

    return len(cand_words & met_words) / max(len(met_words), 1)


def recommend_metiers_fit_first(
    client: OpenAI,
    embed_model: str,
    gap_metier_df: pd.DataFrame,
    candidate_profile: dict,
    region: str,
    department: str,
    top_k: int = 10,
    tie_break_with_gap: bool = False,
    w_fit: float = 0.90,
    w_gap: float = 0.10,
) -> pd.DataFrame:
    """
    Fit-first ranking:
      - Compute fit_score for every metier IN THE WHOLE DATASET (unique metier labels)
      - Take Top K by fit_score
      - Attach local demand_score for the selected region+department (if available)
      - Rank stays by fit unless tie_break_with_gap=True
    """

    # --- 1) Build candidate embedding ---
    cand_text = (
        f"metier: {candidate_profile.get('metier','')}\n"
        f"hard_skills: {', '.join(candidate_profile.get('hard_skills', [])[:60])}\n"
        f"soft_skills: {', '.join(candidate_profile.get('soft_skills', [])[:40])}\n"
        f"summary: {candidate_profile.get('summary','')}"
    )
    cand_emb = get_embedding(client, embed_model, cand_text)

    # --- 2) Fit score for each unique metier label (global) ---
    all_metiers = sorted(gap_metier_df["metier"].dropna().astype(str).unique().tolist())

    cache = _get_metier_emb_cache()
    fit_scores = []
    for m in all_metiers:
        key = f"{embed_model}::metier::{m}"
        if key not in cache:
            cache[key] = get_embedding(client, embed_model, f"metier: {m}")
        emb_fit = cosine(cand_emb, cache[key])

        # Hybrid: add a small skills/label overlap
        overlap = _skill_overlap_score(candidate_profile, m)
        fit = (0.85 * emb_fit) + (0.15 * overlap)

        fit_scores.append((m, fit))

    fit_df = pd.DataFrame(fit_scores, columns=["metier", "fit_score"]).sort_values("fit_score", ascending=False)

    # --- 3) Take top K by fit ---
    top = fit_df.head(top_k).copy()

    # --- 4) Attach local demand (gap) for these metiers ---
    local = gap_metier_df[
        (gap_metier_df["region"] == region) &
        (gap_metier_df["department"] == department)
    ][["metier", "demand_score"]].copy()

    # If local empty, just attach 0
    if local.empty:
        top["demand_score"] = 0.0
        top["demand_rank_in_location"] = None
    else:
        local["demand_score"] = pd.to_numeric(local["demand_score"], errors="coerce").fillna(0.0)

        # demand rank among all jobs in this location
        local = local.sort_values("demand_score", ascending=False)
        local["demand_rank_in_location"] = range(1, len(local) + 1)

        top = top.merge(local, on="metier", how="left")

        top["demand_score"] = top["demand_score"].fillna(0.0)
        top["demand_rank_in_location"] = top["demand_rank_in_location"]

    # --- 5) Optional: tiny tie-break with demand ---
    if tie_break_with_gap:
        # normalize demand in the local context
        if "demand_score" in top.columns:
            d = top["demand_score"].astype(float)
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)
        else:
            d_norm = 0.0

        # normalize fit within top set
        f = top["fit_score"].astype(float)
        f_norm = (f - f.min()) / (f.max() - f.min() + 1e-9)

        top["final_score"] = (w_fit * f_norm) + (w_gap * d_norm)
        top = top.sort_values("final_score", ascending=False)
    else:
        top["final_score"] = top["fit_score"]

    # Pretty ordering
    return top[["metier", "fit_score", "demand_score", "demand_rank_in_location", "final_score"]]


# =========================
# JOB MATCH
# =========================

def job_match_score(cv_profile: dict, job_profile: dict) -> dict:
    cv_hard = set(x.lower() for x in cv_profile.get("hard_skills", []))
    job_hard = set(x.lower() for x in job_profile.get("hard_skills", []))

    cv_soft = set(x.lower() for x in cv_profile.get("soft_skills", []))
    job_soft = set(x.lower() for x in job_profile.get("soft_skills", []))

    hard_overlap = len(cv_hard & job_hard) / max(len(job_hard), 1)
    soft_overlap = len(cv_soft & job_soft) / max(len(job_soft), 1)

    score = 0.7 * hard_overlap + 0.3 * soft_overlap

    missing_hard = [s for s in job_profile.get("hard_skills", []) if s.lower() not in cv_hard]
    missing_soft = [s for s in job_profile.get("soft_skills", []) if s.lower() not in cv_soft]

    matched_hard = [s for s in job_profile.get("hard_skills", []) if s.lower() in cv_hard]
    matched_soft = [s for s in job_profile.get("soft_skills", []) if s.lower() in cv_soft]

    return {
        "score_pct": round(score * 100, 1),
        "hard_overlap": round(hard_overlap * 100, 1),
        "soft_overlap": round(soft_overlap * 100, 1),
        "matched_hard": uniq(matched_hard),
        "missing_hard": uniq(missing_hard),
        "matched_soft": uniq(matched_soft),
        "missing_soft": uniq(missing_soft),
    }

def _fix_mojibake_text(s: str) -> str:
    """
    Fix typical French mojibake like 'mÇ¸tier' -> 'métier'
    (UTF-8 bytes decoded as latin1).
    """
    if not isinstance(s, str):
        return s
    if "Ã" in s or "Ç" in s or "Â" in s or "�" in s:
        try:
            return s.encode("latin1").decode("utf-8")
        except Exception:
            return s
    return s


def _fix_mojibake_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].map(_fix_mojibake_text)
    return df


def read_bmo_csv(uploaded_file) -> pd.DataFrame:
    """
    Robust reader for BMO_2020_2025_merged_clean.csv:
    - tries multiple encodings (utf-8-sig, cp1252, latin1)
    - detects delimiter
    - fixes header misalignment (header=0 vs header=1)
    - drops Unnamed index columns
    - repairs mojibake text
    """
    raw = uploaded_file.getvalue()

    # --- detect separator using a forgiving decode ---
    head_text = raw[:8000].decode("latin1", errors="replace")
    first_nonempty = next((ln for ln in head_text.splitlines() if ln.strip()), "")
    sep = ";" if first_nonempty.count(";") >= first_nonempty.count(",") else ","

    def header_is_suspicious(cols) -> bool:
        cols = [str(c).strip() for c in cols]
        for c in cols[:12]:
            if not c:
                continue
            # suspicious if looks like a date or a pure number
            if c.replace(".", "").replace("-", "").isdigit():
                return True
            try:
                dt = pd.to_datetime(c, errors="raise", dayfirst=True)
                # if it parses to a datetime, it's suspicious as a header
                return True
            except Exception:
                pass
        return False

    # --- try encodings + header row options ---
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    header_opts = [0, 1]

    last_err = None
    for enc in encodings:
        for hdr in header_opts:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=sep,
                    encoding=enc,
                    engine="python",
                    on_bad_lines="skip",
                    header=hdr,
                    quotechar='"',
                    escapechar="\\",
                )

                # Drop index-like columns
                df = df.loc[:, ~df.columns.astype(str).str.lower().str.startswith("unnamed")]

                # Basic sanity: must have at least a few columns
                if df.shape[1] < 5:
                    continue

                # If header looks like values (dates/numbers), try next header option
                if header_is_suspicious(df.columns):
                    continue

                df.columns = df.columns.astype(str).str.strip()
                df = _fix_mojibake_df(df)
                return df

            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"CSV could not be read with common encodings. Last error: {last_err}")


def build_yearly_series(
    df: pd.DataFrame,
    year_col: str,
    target_col: str,
    group_cols: list[str],
    agg: str = "sum",
) -> pd.DataFrame:
    work = df.copy()

    # Robust year extraction (works even if year_col is a date string)
    work["_year"] = to_year_series(work[year_col])
    work = work.dropna(subset=["_year"])
    work["_year"] = work["_year"].astype(int)

    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col])

    keys = (group_cols + ["_year"]) if group_cols else ["_year"]

    if agg == "mean":
        out = work.groupby(keys, dropna=False)[target_col].mean().reset_index()
    else:
        out = work.groupby(keys, dropna=False)[target_col].sum().reset_index()

    out = out.rename(columns={"_year": "year", target_col: "y"})
    return out.sort_values((group_cols + ["year"]) if group_cols else ["year"])


def linear_forecast_per_group(
    panel: pd.DataFrame,
    group_cols: list[str],
    horizon_years: int = 5,
    min_points: int = 4,
    holdout_years: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pure NumPy linear trend per group:
      y = a*year + b
    Returns:
      forecast_df: history + forecast with yhat
      metrics_df: MAE/RMSE on last N years (if possible)
    """
    if group_cols:
        groups = panel.groupby(group_cols, dropna=False)
    else:
        groups = [(("ALL",), panel.copy())]

    all_out = []
    all_metrics = []

    for g_key, g in groups:
        g = g.dropna(subset=["year", "y"]).sort_values("year")
        years = g["year"].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=float)

        uniq_years = np.unique(years.astype(int))
        if len(uniq_years) < min_points:
            continue

        # Holdout split by last N unique years
        if holdout_years >= 1 and len(uniq_years) > holdout_years:
            cutoff = np.sort(uniq_years)[-holdout_years]
            train_mask = years < cutoff
            test_mask = years >= cutoff
        else:
            train_mask = np.ones_like(years, dtype=bool)
            test_mask = np.zeros_like(years, dtype=bool)

        X_train = years[train_mask]
        y_train = y[train_mask]
        if len(X_train) < 2:
            continue

        # Fit y = a*year + b
        a, b = np.polyfit(X_train, y_train, deg=1)

        # Evaluate
        if test_mask.any():
            y_pred = a * years[test_mask] + b
            y_true = y[test_mask]
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        else:
            mae = np.nan
            rmse = np.nan

        last_year = int(np.max(uniq_years))
        future_years = np.arange(last_year + 1, last_year + 1 + horizon_years, dtype=int)
        y_future = a * future_years + b
        y_future = np.maximum(y_future, 0)  # prevent negative counts

        # history rows
        hist = g.copy()
        hist["yhat"] = np.nan
        hist["is_forecast"] = False

        # forecast rows
        fore = pd.DataFrame({"year": future_years, "y": np.nan, "yhat": y_future, "is_forecast": True})
        if group_cols:
            if not isinstance(g_key, tuple):
                g_key = (g_key,)
            for i, c in enumerate(group_cols):
                fore[c] = g_key[i]
            fore = fore[[*group_cols, "year", "y", "yhat", "is_forecast"]]

        out = pd.concat([hist, fore], ignore_index=True)

        # metrics row
        m = {"n_years": int(len(uniq_years)), "mae": mae, "rmse": rmse}
        if group_cols:
            if not isinstance(g_key, tuple):
                g_key = (g_key,)
            for i, c in enumerate(group_cols):
                m[c] = g_key[i]
        all_metrics.append(m)
        all_out.append(out)

    forecast_df = pd.concat(all_out, ignore_index=True) if all_out else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
    return forecast_df, metrics_df

def clean_columns_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    # Drop "Unnamed: 0" / index columns
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]

    return df


def try_repair_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    If header seems broken, try to re-read by assuming:
    - first row contains headers
    In Streamlit upload context we can't easily re-read without the buffer,
    so this is only used if you read from path. In upload mode, we instead
    show a clear error + instruction.
    """
    return df  # placeholder (see UI section below for upload-safe approach)


def year_likeness_score(s: pd.Series) -> float:
    """
    Returns a score [0,1] showing how likely a column is a year/time column.
    Works for:
      - numeric year columns (e.g., 2020, 2021)
      - date columns stored as strings (e.g., 1.01.2021, 2021-01-01)
    """
    if s is None or len(s) == 0:
        return 0.0

    # Try numeric years
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().sum() >= max(10, int(0.1 * len(s))):
        frac_yearlike = ((sn >= 1990) & (sn <= 2100)).mean()
        if frac_yearlike >= 0.3:
            return float(frac_yearlike)

    # Try parsing as dates (dayfirst helps FR)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if dt.notna().sum() >= max(10, int(0.1 * len(s))):
        years = dt.dt.year
        frac_years_ok = ((years >= 1990) & (years <= 2100)).mean()
        return float(frac_years_ok)

    return 0.0


def find_year_candidates(df: pd.DataFrame) -> list[str]:
    """
    Return columns sorted by year-likeness score, high to low.
    """
    scores = []
    for c in df.columns:
        score = year_likeness_score(df[c])
        scores.append((c, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    # keep anything with a minimal score; but always return top few for manual choice
    candidates = [c for c, sc in scores if sc >= 0.25]
    if not candidates:
        candidates = [c for c, _ in scores[:8]]  # let user choose among top 8 likely
    return candidates


def to_year_series(s: pd.Series) -> pd.Series:
    """
    Convert any time-ish series to integer year.
    """
    sn = pd.to_numeric(s, errors="coerce")
    # If mostly year-like numeric, use it
    if sn.notna().sum() >= max(10, int(0.1 * len(s))) and ((sn >= 1990) & (sn <= 2100)).mean() >= 0.3:
        return sn.round().astype("Int64")

    # Else parse dates
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt.dt.year.astype("Int64")


def find_numeric_candidates(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    numeric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(10, int(0.2 * len(df))):
            numeric_cols.append(c)
    return numeric_cols


def pick_default(cols: list[str], preferred: list[str]) -> int:
    """
    Return index for selectbox default
    """
    low_map = {str(c).lower(): i for i, c in enumerate(cols)}
    for p in preferred:
        if p.lower() in low_map:
            return low_map[p.lower()]
    return 0

def detect_bmo_dimensions(df: pd.DataFrame) -> dict:
    """
    Try to map your dataset columns to Region/Department/Sector labels.
    Adjust the candidate lists if your column names differ.
    """
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    region_col = pick(["NOM_REG", "Nom_reg", "region", "Region"])
    dept_col = pick(["NomDept", "NOM_DEP", "department", "Departement", "Département"])
    sector_col = pick(["Lbl_fam_met", "sector", "Secteur", "Domaine", "Famille", "Lbl_fam"])

    return {"region": region_col, "department": dept_col, "sector": sector_col}


def safe_sorted_unique(df: pd.DataFrame, col: str) -> list[str]:
    if not col or col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    return sorted(vals.unique().tolist())

def recommend_training_plan(
    client: OpenAI,
    model: str,
    candidate_profile: dict,
    recommended_metiers: list[str],
    gap_metier_df: pd.DataFrame,
    region: str,
    department: str,
    ui_lang: str = "en",
    top_n_trainings: int = 10,
) -> dict:
    """
    Generate training advice in the selected UI language ("en" or "fr"),
    based on the candidate profile + the top recommended métiers.

    Returns strict JSON with:
      - priority_trainings: list[{topic, why, recommended_for_metiers, suggested_format, estimated_effort, portfolio_proof}]
      - quick_wins: list[str]
      - cv_edits: list[str]
    """

    language_line = "French" if ui_lang == "fr" else "English"

    # Attach local demand context as optional input (not controlling ranking)
    demand_context = []
    try:
        if not gap_metier_df.empty and {"region", "department", "metier"}.issubset(set(gap_metier_df.columns)):
            local = gap_metier_df[
                (gap_metier_df["region"] == region) &
                (gap_metier_df["department"] == department)
            ][["metier", "demand_score"]].copy()

            if not local.empty:
                local["demand_score"] = pd.to_numeric(local["demand_score"], errors="coerce").fillna(0.0)
                for m in (recommended_metiers or [])[:12]:
                    hit = local[local["metier"] == m]
                    demand_context.append({
                        "metier": m,
                        "demand_score": float(hit["demand_score"].iloc[0]) if not hit.empty else None
                    })
    except Exception:
        # keep demand_context empty if anything goes wrong
        demand_context = []

    payload = {
        "candidate": {
            "metier": candidate_profile.get("metier", ""),
            "hard_skills": (candidate_profile.get("hard_skills", []) or [])[:80],
            "soft_skills": (candidate_profile.get("soft_skills", []) or [])[:50],
            "summary": candidate_profile.get("summary", ""),
        },
        "recommended_metiers": (recommended_metiers or [])[:12],
        "location": {"region": region, "department": department},
        "local_demand_context": demand_context,
        "constraints": {
            "max_trainings": top_n_trainings,
            "style": "practical, job-search focused, France market aware",
            "time_horizon": "2–8 weeks",
        },
    }

    schema = {
        "type": "object",
        "properties": {
            "priority_trainings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "why": {"type": "string"},
                        "recommended_for_metiers": {"type": "array", "items": {"type": "string"}},
                        "suggested_format": {"type": "string"},
                        "estimated_effort": {"type": "string"},
                        "portfolio_proof": {"type": "string"},
                    },
                    "required": [
                        "topic",
                        "why",
                        "recommended_for_metiers",
                        "suggested_format",
                        "estimated_effort",
                        "portfolio_proof",
                    ],
                    "additionalProperties": False,
                },
            },
            "quick_wins": {"type": "array", "items": {"type": "string"}},
            "cv_edits": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["priority_trainings", "quick_wins", "cv_edits"],
        "additionalProperties": False,
    }

    resp = client.responses.create(
        model=model,
        instructions=(
            f"Write everything in {language_line}.\n\n"
            "You are a career advisor for tech/data/business roles in France.\n"
            "Given a candidate profile and a list of recommended métiers, propose a training plan.\n\n"
            "Rules:\n"
            "- Be specific (tools, methods, certifications, projects).\n"
            "- Avoid generic advice like 'improve communication'.\n"
            "- Each training item MUST include a 'portfolio_proof' (what to build to prove it).\n"
            "- Keep effort realistic for 2–8 weeks.\n"
            "- Do NOT invent employer-specific requirements; keep it general but concrete.\n"
            "- Return JSON only (no markdown, no extra text).\n"
        ),
        input=json.dumps(payload),
        text={
            "format": {
                "type": "json_schema",
                "name": "training_plan",
                "schema": schema,
                "strict": True,
            }
        },
    )

    return json.loads(resp.output_text)
    
# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Talent Recommendations + Job Match", layout="wide")
# =========================
# GLOBAL LANGUAGE SELECTOR
# =========================

LANG_OPTIONS = {"English": "en", "Français": "fr"}

if "UI_LANG" not in st.session_state:
    st.session_state["UI_LANG"] = "en"

selected_label = st.sidebar.selectbox(
    "Language / Langue",
    list(LANG_OPTIONS.keys()),
    index=list(LANG_OPTIONS.values()).index(st.session_state["UI_LANG"]),
    key="global_language_selector"
)

st.session_state["UI_LANG"] = LANG_OPTIONS[selected_label]
UI_LANG = st.session_state["UI_LANG"]
st.title(t("app_title", UI_LANG))

gap_metier_df, gap_sector_df = load_gap_data()

st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Paste your key here for hackathon demo.")
model = st.sidebar.text_input("LLM model", value=DEFAULT_MODEL)
embed_model = st.sidebar.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)

if not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to enable extraction & recommendations.")
    st.stop()

client = get_openai_client(api_key)

regions = sorted(gap_metier_df["region"].dropna().unique().tolist())
region = st.selectbox("Region", regions)

deps = sorted(gap_metier_df[gap_metier_df["region"] == region]["department"].dropna().unique().tolist())
department = st.selectbox("City / Department", deps)

tab1, tab2, tab3 = st.tabs([
    t("tab_reco", UI_LANG),
    t("tab_match", UI_LANG),
    t("tab_forecast", UI_LANG),
])



with tab1:
    st.subheader("Recommendations based on your location demand gaps + your qualifications")

    colA, colB = st.columns([1, 1], gap="large")

    with colA:
        st.markdown("### 1) Upload CV (to understand qualifications)")
        cv_file = st.file_uploader("CV file", type=["pdf", "docx", "txt"], key="cv_reco")
        st.caption("We extract hard skills, soft skills, and a predicted métier from your CV.")

    with colB:
        st.markdown("### 2) Recommendation weights")
        w_gap = st.slider("Weight: demand gap (BMO)", 0.0, 1.0, 0.55, 0.05)
        w_fit = 1.0 - w_gap
        st.write(f"Weight: profile fit = {w_fit:.2f}")

        top_k = st.slider("How many recommendations?", 3, 15, 10, 1)

    if cv_file:
        cv_text = read_uploaded_text(cv_file)

        with st.spinner("Extracting candidate profile..."):
            cand = extract_profile(
                client,
                model,
                cv_text,
                st.session_state["UI_LANG"]
            )

        st.markdown("### Extracted Candidate Skills")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Predicted métier:** {cand.get('metier','—')}")
            render_bullets("Hard skills", uniq(cand.get("hard_skills", [])))
        with c2:
            render_bullets("Soft skills", uniq(cand.get("soft_skills", [])))
            st.markdown("#### Summary")
            st.write(cand.get("summary", "—"))

        st.divider()

        with st.spinner("Generating recommendations (gap + fit)..."):
            rec_df = recommend_metiers_fit_first(
                client=client,
                embed_model=embed_model,
                gap_metier_df=gap_metier_df,
                candidate_profile=cand,
                region=region,
                department=department,
                top_k=top_k,
                tie_break_with_gap=False,  # keep fit-only ranking
            )

        st.markdown("### Recommended métiers for this location")
        if rec_df.empty:
            st.info("No demand data found for this location.")
        else:
            show = rec_df.copy()
            show["fit_score"] = show["fit_score"].round(3)
            show["final_score"] = show["final_score"].round(3)
            show["demand_score"] = show["demand_score"].round(4)
            st.dataframe(show, use_container_width=True)
            st.divider()
st.markdown("### 🎓 Training Advice (to improve your fit)")

with st.expander("Generate a training plan based on my CV + the recommended métiers", expanded=True):
    train_btn = st.button("Generate training advice", key="tab1_train_btn")

    if train_btn:
        # Use the fit-first recommendations list (metier labels)
        metiers_for_training = rec_df["metier"].astype(str).tolist() if not rec_df.empty else []

        if not metiers_for_training:
            st.info("No recommended métiers available yet.")
        else:
            with st.spinner("Creating training plan..."):
                plan = recommend_training_plan(
                    client=client,
                    model=model,
                    candidate_profile=cand,
                    recommended_metiers=metiers_for_training,
                    gap_metier_df=gap_metier_df,
                    region=region,
                    department=department,
                    ui_lang=st.session_state["UI_LANG"],
                )

            # Display nicely
            st.markdown("#### Priority trainings")
            for i, item in enumerate(plan["priority_trainings"], start=1):
                st.markdown(f"**{i}. {item['topic']}**")
                st.write(item["why"])
                st.caption(
                    f"Recommended for: {', '.join(item['recommended_for_metiers'][:4])} | "
                    f"Format: {item['suggested_format']} | "
                    f"Effort: {item['estimated_effort']}"
                )
                st.write(f"**Portfolio proof:** {item['portfolio_proof']}")
                st.write("")

            st.markdown("#### Quick wins (1–3 days)")
            for q in plan["quick_wins"]:
                st.write(f"• {q}")

            st.markdown("#### CV edits (to reflect these trainings)")
            for cved in plan["cv_edits"]:
                st.write(f"• {cved}")
            st.caption("Ranking is CV-fit first. Demand gap is shown only as context for the selected location.")

        if not rec_df.empty:
            st.divider()
            st.markdown("### Why these are recommended (Top 3)")
            top3 = rec_df.head(3)["metier"].tolist()

            expl_prompt = {
                "candidate": {
                    "metier": cand.get("metier"),
                    "hard_skills": cand.get("hard_skills", [])[:40],
                    "soft_skills": cand.get("soft_skills", [])[:20],
                    "summary": cand.get("summary", "")
                },
                "location": {"region": region, "department": department},
                "recommended_metiers": top3
            }

            schema = {
                "type": "object",
                "properties": {
                    "reasons": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "metier": {"type": "string"},
                                "reason": {"type": "string"}
                            },
                            "required": ["metier", "reason"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["reasons"],
                "additionalProperties": False
            }

            resp = client.responses.create(
                model=model,
                instructions=(
                    "Explain in 1-2 sentences per métier why it fits the candidate profile AND the local demand gap.\n"
                    "Return JSON only."
                ),
                input=json.dumps(expl_prompt),
                text={"format": {"type": "json_schema", "name": "reco_reasons", "schema": schema, "strict": True}}
            )
            reasons = json.loads(resp.output_text)["reasons"]
            for r in reasons:
                st.write(f"**{r['metier']}** — {r['reason']}")


with tab2:
    st.subheader("Match a CV with a specific Job Description (separate from recommendations)")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Upload CV")
        cv_file2 = st.file_uploader("CV file", type=["pdf", "docx", "txt"], key="cv_match")

    with right:
        st.markdown("### Paste Job Description")
        jd_text = st.text_area("Job Description", height=260, key="jd_match")

    if cv_file2 and jd_text.strip():
        cv_text2 = read_uploaded_text(cv_file2)

        with st.spinner("Extracting CV profile..."):
            cv_prof = extract_profile(client, model, cv_text2, st.session_state["UI_LANG"])

        with st.spinner("Extracting JD profile..."):
            jd_prof = extract_profile(client, model, jd_text, st.session_state["UI_LANG"])

        metier_for_demand = jd_prof.get("metier") or cv_prof.get("metier") or ""
        demand_score, demand_source, matched_metier = demand_lookup(
            gap_metier_df, metier_for_demand, region, department
        )

        match = job_match_score(cv_prof, jd_prof)

        st.markdown(f"## Match Score: **{match['score_pct']}%**")
        st.caption(f"Hard overlap: {match['hard_overlap']}% | Soft overlap: {match['soft_overlap']}%")

        st.markdown("### Demand Priority (for this location)")
        st.write(f"Metier (model): {metier_for_demand}")
        st.write(f"Matched metier (data): {matched_metier or '—'}")
        st.write(f"Demand score: {demand_score}")
        st.write(f"Match type: {demand_source}")

        st.divider()

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("### CV Extracted Skills")
            render_bullets("Hard skills", uniq(cv_prof.get("hard_skills", [])))
            render_bullets("Soft skills", uniq(cv_prof.get("soft_skills", [])))

        with c2:
            st.markdown("### JD Extracted Skills")
            render_bullets("Hard skills", uniq(jd_prof.get("hard_skills", [])))
            render_bullets("Soft skills", uniq(jd_prof.get("soft_skills", [])))

        st.divider()

        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.markdown("### Matched Skills")
            render_bullets("Matched hard", match["matched_hard"])
            render_bullets("Matched soft", match["matched_soft"])
        with c4:
            st.markdown("### Missing Skills")
            render_bullets("Missing hard", match["missing_hard"])
            render_bullets("Missing soft", match["missing_soft"])

    else:
        st.info("Upload a CV and paste a Job Description to see the job match results.")


# =========================
# TAB 3 — PREDICTIVE MODELING (BMO FORECAST)
# Clean, state-safe, production-ready
# =========================

with tab3:
    st.subheader("📈 Predictive Modeling (BMO 2020–2025) — Forecast by Region / Department / Sector")

    # -------------------------
    # Helpers (Tab-local safe)
    # -------------------------
    def _norm_key(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
            .str.replace("\u00a0", " ", regex=False)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )

    def _safe_sorted_unique(df: pd.DataFrame, col: str) -> list[str]:
        if col not in df.columns:
            return []
        vals = df[col].dropna().astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
        vals = vals[vals != ""]
        return sorted(vals.unique().tolist())

    def _year_list(df: pd.DataFrame, year_col: str) -> list[int]:
        yrs = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int).unique().tolist()
        return sorted(yrs)

    def _build_yearly_series(df: pd.DataFrame, year_col: str, target_col: str, agg: str) -> pd.DataFrame:
        # Returns columns: year, y
        work = df.copy()
        work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
        work = work.dropna(subset=[year_col, target_col])

        work[year_col] = work[year_col].astype(int)

        if agg == "mean":
            out = work.groupby(year_col, dropna=False)[target_col].mean().reset_index()
        else:
            out = work.groupby(year_col, dropna=False)[target_col].sum().reset_index()

        out = out.rename(columns={year_col: "year", target_col: "y"}).sort_values("year")
        return out

    def _linear_forecast(panel: pd.DataFrame, horizon_years: int) -> pd.DataFrame:
        """
        panel: year,y sorted
        returns history + forecast with yhat, is_forecast
        """
        panel = panel.sort_values("year")
        years = panel["year"].to_numpy(dtype=float)
        y = panel["y"].to_numpy(dtype=float)

        uniq = np.unique(years.astype(int))
        n_years = len(uniq)

        hist = panel.copy()
        hist["yhat"] = np.nan
        hist["is_forecast"] = False

        last_year = int(np.max(uniq))
        future_years = np.arange(last_year + 1, last_year + 1 + horizon_years, dtype=int)

        # 1 year => flat forecast
        if n_years == 1:
            last_val = float(y[-1])
            yhat = np.array([last_val] * horizon_years, dtype=float)
            fore = pd.DataFrame({"year": future_years, "y": np.nan, "yhat": yhat, "is_forecast": True})
            return pd.concat([hist, fore], ignore_index=True)

        # 2+ years => linear fit (polyfit)
        a, b = np.polyfit(years, y, deg=1)
        yhat = a * future_years + b
        yhat = np.maximum(yhat, 0)  # avoid negative counts

        fore = pd.DataFrame({"year": future_years, "y": np.nan, "yhat": yhat, "is_forecast": True})
        return pd.concat([hist, fore], ignore_index=True)

    # -------------------------
    # Upload + Read (state-safe)
    # -------------------------
    uploaded = st.file_uploader("Upload BMO CSV (2020–2025)", type=["csv"], key="tab3_upload_bmo")

    if not uploaded:
        st.info("Upload the BMO CSV to enable forecasting.")
        st.stop()

    # Read via your robust reader (must exist in app.py)
    # Recommended: use the fixed CSV you generated OR the robust multi-encoding reader.
    try:
        df_bmo = read_bmo_csv(uploaded)  # <- uses multi-encoding + header repair
    except Exception as e:
        st.error(f"CSV could not be read: {e}")
        st.stop()

    # Optional: ensure no duplicated columns
    df_bmo = df_bmo.loc[:, ~df_bmo.columns.duplicated()]

    # -------------------------
    # Column mapping (BMO)
    # -------------------------
    # Expect these to exist; adapt here if your names differ
    required = {
        "year": "annee",
        "target": "met",
        "region": "NOM_REG",
        "department": "NomDept",
        "sector": "Lbl_fam_met",
    }

    missing = [k for k, c in required.items() if c not in df_bmo.columns]
    if missing:
        st.error(
            "Missing required columns for Region/Department/Sector forecasting.\n\n"
            f"Missing mappings: {missing}\n\n"
            f"Expected columns: {list(required.values())}"
        )
        st.write("Detected columns:", df_bmo.columns.tolist()[:60])
        st.stop()

    year_col = required["year"]
    target_col = required["target"]
    region_col = required["region"]
    dept_col = required["department"]
    sector_col = required["sector"]

    # Create stable keys (avoids tiny label changes across years)
    df_bmo["_REG_KEY"] = _norm_key(df_bmo[region_col])
    df_bmo["_DEP_KEY"] = _norm_key(df_bmo[dept_col])
    df_bmo["_SEC_KEY"] = _norm_key(df_bmo[sector_col])

    # -------------------------
    # Controls
    # -------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        agg = st.selectbox("Aggregation", ["sum", "mean"], index=0, key="tab3_agg")
    with c2:
        horizon = st.number_input("Forecast horizon (years)", 1, 10, 5, 1, key="tab3_horizon")
    with c3:
        show_preview = st.checkbox("Show filtered rows preview", value=False, key="tab3_preview")
    with c4:
        auto_run = st.checkbox("Auto-run forecast on selection", value=True, key="tab3_autorun")

    st.divider()
    st.markdown("### Select Region / Department / Sector")

    # -------------------------
    # Cascading selectors (state-safe keys)
    # -------------------------
    regions = _safe_sorted_unique(df_bmo, region_col)
    if not regions:
        st.error("No regions found in the dataset.")
        st.stop()

    selected_region = st.selectbox("Region", regions, key="tab3_region")
    reg_key = selected_region.strip().lower()

    df_r = df_bmo[df_bmo["_REG_KEY"] == reg_key]

    depts = _safe_sorted_unique(df_r, dept_col)
    if not depts:
        st.error("No departments found for the selected region.")
        st.stop()

    selected_dept = st.selectbox("Department", depts, key="tab3_department")
    dep_key = selected_dept.strip().lower()

    df_d = df_r[df_r["_DEP_KEY"] == dep_key]

    # Make sector selector year-aware to avoid selections with tiny year coverage
    # Compute year coverage per sector for this Region+Dept
    cov = (
        df_d.assign(_Y=pd.to_numeric(df_d[year_col], errors="coerce"))
            .dropna(subset=["_Y"])
            .groupby("_SEC_KEY")["_Y"]
            .nunique()
            .reset_index(name="n_years")
    )

    # Build mapping from key->pretty label (first occurrence)
    key_to_label = (
        df_d[["_SEC_KEY", sector_col]]
        .dropna()
        .astype(str)
        .groupby("_SEC_KEY")[sector_col]
        .first()
        .to_dict()
    )

    cov["sector_label"] = cov["_SEC_KEY"].map(key_to_label)
    cov = cov.dropna(subset=["sector_label"]).sort_values(["n_years", "sector_label"], ascending=[False, True])

    min_years_required = st.slider("Minimum years required (for reliable trend)", 1, 5, 3, 1, key="tab3_min_years")

    valid_cov = cov[cov["n_years"] >= min_years_required]
    if valid_cov.empty:
        st.warning(
            "No sector in this Region+Department meets the minimum year requirement.\n\n"
            "Lower the 'Minimum years required' slider or pick a different Region/Department."
        )
        # Still allow user to pick from all sectors (fallback)
        sectors_all = _safe_sorted_unique(df_d, sector_col)
        if not sectors_all:
            st.stop()
        selected_sector = st.selectbox("Sector", sectors_all, key="tab3_sector_fallback")
        sec_key = selected_sector.strip().lower()
        df_slice = df_d[df_d["_SEC_KEY"] == sec_key]
        sector_years = _year_list(df_slice, year_col)
    else:
        sector_labels = valid_cov["sector_label"].tolist()
        selected_sector = st.selectbox("Sector", sector_labels, key="tab3_sector")
        # find its key
        sec_key = None
        for k, v in key_to_label.items():
            if v == selected_sector:
                sec_key = k
                break
        df_slice = df_d[df_d["_SEC_KEY"] == sec_key] if sec_key is not None else df_d.iloc[0:0]
        sector_years = _year_list(df_slice, year_col)

    st.info(
        f"Selected → {region_col}: {selected_region} | {dept_col}: {selected_dept} | {sector_col}: {selected_sector}\n\n"
        f"Rows in slice: {len(df_slice):,} | Years available: {sector_years} (count: {len(sector_years)})"
    )

    if show_preview:
        st.dataframe(df_slice.head(50), use_container_width=True)

    # -------------------------
    # Forecast execution (button or auto-run)
    # -------------------------
    run = st.button("Run Forecast", key="tab3_run") if not auto_run else True

    if run:
        panel = _build_yearly_series(df_slice, year_col=year_col, target_col=target_col, agg=agg)

        if panel.empty:
            st.warning("No usable time series after filtering (year/target missing). Try a different selection.")
            st.stop()

        # Always forecast (flat if only 1 year, linear if 2+ years)
        forecast_df = _linear_forecast(panel, horizon_years=int(horizon))

        st.markdown("### Forecast Output (Selected Slice)")
        st.dataframe(forecast_df.sort_values("year"), use_container_width=True)

        chart_df = forecast_df[["year", "y", "yhat"]].set_index("year").sort_index()
        st.line_chart(chart_df, use_container_width=True)

        # Optional: simple year-over-year growth summary (history only)
        hist = panel.sort_values("year").copy()
        if len(hist) >= 2:
            hist["yoy_pct"] = hist["y"].pct_change() * 100
            st.markdown("### Historical YoY Change (History Only)")
            st.dataframe(hist, use_container_width=True)