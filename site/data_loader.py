"""讀取 exports/analytical/*.csv，去除個資欄位後回傳 DataFrame。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "exports" / "analytical"

PII_COLS = {"姓名", "student_id", "報名序號", "統測報名序號", "統一入學測驗報名序號", "座號"}


def _drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c in PII_COLS], errors="ignore")


@st.cache_data
def load_students() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "00_students.csv", dtype={"班級_標準": str, "座號": str})
    return _drop_pii(df)


@st.cache_data
def load_wide() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "01_wide_scores.csv", dtype={"班級_標準": str})
    return _drop_pii(df)


@st.cache_data
def load_admissions() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "03_admissions.csv", dtype={"班級_標準": str})
    return _drop_pii(df)


@st.cache_data
def load_thresholds() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "05_分發門檻_商管群.csv")
    return df
