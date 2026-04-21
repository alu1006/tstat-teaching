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


@st.cache_data
def load_tuijian_stats() -> pd.DataFrame:
    """推甄（科大甄選）錄取學生的加權統測分統計（不含個人資料）。
    用各校加權公式換算分數；以學校×科系分類彙整，至少 2 人才顯示。
    """
    import re as _re

    _pat = _re.compile(r"(國文|英文|數學|專業\(一\)|專業\(二\))\*(\d+\.?\d*)")
    _key = {"國文":"統測_國文分數","英文":"統測_英文分數","數學":"統測_數學B分數",
            "專業(一)":"統測_專一分數","專業(二)":"統測_專二分數"}

    def _weighted(row):
        try:
            return sum(row[_key[k]] * float(v) for k, v in _pat.findall(row["各科目加權"]))
        except Exception:
            return None

    app  = pd.read_csv(DATA_DIR / "04_applications.csv")
    wide = pd.read_csv(DATA_DIR / "01_wide_scores.csv", dtype={"班級_標準": str})
    th   = pd.read_csv(DATA_DIR / "05_分發門檻_商管群.csv")

    accepted = app[app["是否最終錄取"] == True][
        ["student_id","畢業年度","志願學校","志願科系分類"]
    ]
    score_cols = ["student_id","畢業年度",
                  "統測_國文分數","統測_英文分數","統測_數學B分數",
                  "統測_專一分數","統測_專二分數"]
    merged = accepted.merge(wide[score_cols], on=["student_id","畢業年度"], how="inner")

    # 取各校最常見加權公式（同學校不同科系公式可能不同，以 mode 代表）
    th_formula = (
        th.groupby(["年度","學校名稱"])["各科目加權"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )
    merged = merged.merge(
        th_formula, left_on=["畢業年度","志願學校"],
        right_on=["年度","學校名稱"], how="left"
    )

    merged["加權分"] = merged.apply(_weighted, axis=1)

    stats = (
        merged.groupby(["畢業年度","志願學校","志願科系分類"])["加權分"]
        .agg(推甄人數="count", 推甄最低加權分="min", 推甄平均加權分="mean")
        .reset_index()
        .rename(columns={"畢業年度":"年度","志願學校":"學校名稱","志願科系分類":"科系分類"})
    )
    # 少於 2 人不顯示分數（防止個資還原）
    mask = stats["推甄人數"] < 2
    stats.loc[mask, ["推甄最低加權分","推甄平均加權分"]] = None
    stats["推甄最低加權分"] = stats["推甄最低加權分"].round(1)
    stats["推甄平均加權分"] = stats["推甄平均加權分"].round(1)
    return stats
