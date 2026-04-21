"""資訊 × 數學 共同教學：三屆學長姐統測資料互動課。

啟動：streamlit run site/app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

import re

from data_loader import load_admissions, load_students, load_thresholds, load_wide

st.set_page_config(page_title="統測資料互動課", layout="wide")

_WEIGHT_PAT = re.compile(r"(國文|英文|數學|專業\(一\)|專業\(二\))\*(\d+\.?\d*)")
_SUBJ_KEY = {"國文": "國文", "英文": "英文", "數學": "數學", "專業(一)": "專一", "專業(二)": "專二"}


def parse_weights(formula: str) -> dict[str, float]:
    """將加權公式字串解析成 {科目: 權重} dict。"""
    return {_SUBJ_KEY[k]: float(v) for k, v in _WEIGHT_PAT.findall(formula)}


def calc_weighted(scores: dict[str, float], weights: dict[str, float]) -> float:
    """依各科分數與該校權重計算加權總分。"""
    return sum(scores.get(s, 0) * w for s, w in weights.items())

DEPT_MAP = {"1": "商經科", "2": "國貿科", "4": "資處科"}
DEPT_ORDER = ["商經科", "國貿科", "資處科"]


def add_dept(df: pd.DataFrame) -> pd.DataFrame:
    """依班級_標準第一碼加上科別欄，並篩掉不在 DEPT_MAP 的班級。"""
    df = df.copy()
    df["科別"] = df["班級_標準"].astype(str).str[0].map(DEPT_MAP)
    return df[df["科別"].notna()].reset_index(drop=True)

# ────────────────────────────────────────────────────────────────
# Section 0 開場
# ────────────────────────────────────────────────────────────────
st.title("三屆學長姐的統測資料")
st.caption("資訊 × 數學 共同課｜所有圖表都是 Python 現場算出來的，改幾行就會跟著變")

students = add_dept(load_students())
wide = add_dept(load_wide())
adm = add_dept(load_admissions())

total_students = len(students)
升學人數 = adm[adm["錄取學校"].notna()]["科別"].count()
平均統測 = wide["統測_總分數"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("三屆總人數", f"{total_students:,}")
c2.metric("有升學紀錄人數", f"{升學人數:,}")
c3.metric("升學率", f"{升學人數 / total_students:.0%}")
c4.metric("統測總分平均", f"{平均統測:.1f}" if pd.notna(平均統測) else "—")

# 各科平均與中位數
SUBJ_COLS = {
    "國文": "統測_國文分數",
    "英文": "統測_英文分數",
    "數學": "統測_數學B分數",
    "專一": "統測_專一分數",
    "專二": "統測_專二分數",
}
rows = []
for dept in DEPT_ORDER + ["全體"]:
    sub = wide if dept == "全體" else wide[wide["科別"] == dept]
    row = {"科別": dept}
    for name, col in SUBJ_COLS.items():
        s = sub[col].dropna()
        row[f"{name}平均"] = round(s.mean(), 1) if len(s) else None
        row[f"{name}中位數"] = round(s.median(), 1) if len(s) else None
    rows.append(row)

summary_df = pd.DataFrame(rows).set_index("科別")

st.subheader("各科統測分數概況")
avg_cols  = [f"{n}平均"  for n in SUBJ_COLS]
med_cols  = [f"{n}中位數" for n in SUBJ_COLS]

tab_avg, tab_med = st.tabs(["平均分數", "中位數"])
with tab_avg:
    st.dataframe(summary_df[avg_cols].style.format("{:.1f}"), width="stretch")
with tab_med:
    st.dataframe(summary_df[med_cols].style.format("{:.1f}"), width="stretch")

st.divider()

# ────────────────────────────────────────────────────────────────
# Section 1 班級錄取地圖
# ────────────────────────────────────────────────────────────────
st.header("① 科別錄取地圖")
st.caption("數學概念：敘述統計（計數、占比）")

sel_dept = st.selectbox("選一個科別看他們都上哪種學校", ["全部"] + DEPT_ORDER)

sub_adm = adm if sel_dept == "全部" else adm[adm["科別"] == sel_dept]

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("學校類型分佈")
    if "學校類型" in sub_adm.columns:
        counts = sub_adm["學校類型"].value_counts().reset_index()
        counts.columns = ["學校類型", "人次"]
        fig = px.bar(counts, x="學校類型", y="人次", color="學校類型", text="人次")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

with col_b:
    st.subheader("科系分類分佈")
    if "科系分類" in sub_adm.columns:
        counts = sub_adm["科系分類"].value_counts().reset_index()
        counts.columns = ["科系分類", "人次"]
        fig = px.bar(counts, x="科系分類", y="人次", color="科系分類", text="人次")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

st.divider()

# ────────────────────────────────────────────────────────────────
# Section 2 模擬考能預測統測嗎？
# ────────────────────────────────────────────────────────────────
st.header("② 模擬考能預測統測嗎？")
st.caption("數學概念：散佈圖、相關係數 r、最小平方迴歸、殘差平方和 SSE")

col_ctrl1, col_ctrl2 = st.columns([1, 2])
exam = col_ctrl1.slider("用第幾次模擬考預測？", 1, 5, 5)
year_sel = col_ctrl2.multiselect(
    "畢業年度", [112, 113, 114], default=[112, 113, 114]
)

x_col, y_col = f"模{exam}_總分數", "統測_總分數"
sub = wide[wide["畢業年度"].isin(year_sel)][[x_col, y_col]].dropna()

if len(sub) < 5:
    st.warning("符合條件的資料太少，請調整篩選。")
else:
    r_val = sub.corr().iloc[0, 1]

    ols = LinearRegression().fit(sub[[x_col]], sub[y_col])
    m_ols, b_ols = float(ols.coef_[0]), float(ols.intercept_)

    with st.expander("✏️ 自己畫一條預測線試試（數學課延伸）", expanded=True):
        st.caption("拉兩個點 → 兩點決定一條直線 y = m·x + b，看看能不能比 OLS 更小的 SSE？")
        x_min, x_max = int(sub[x_col].min()), int(sub[x_col].max())
        y_min, y_max = int(sub[y_col].min()), int(sub[y_col].max())
        cc1, cc2 = st.columns(2)
        p1x = cc1.slider("第 1 點 x（模考總分）", x_min, x_max, x_min + 50, key="p1x")
        p1y = cc1.slider("第 1 點 y（統測總分）", y_min, y_max, y_min + 50, key="p1y")
        p2x = cc2.slider("第 2 點 x", x_min, x_max, x_max - 50, key="p2x")
        p2y = cc2.slider("第 2 點 y", y_min, y_max, y_max - 50, key="p2y")

        m_user = (p2y - p1y) / max(p2x - p1x, 1e-6)
        b_user = p1y - m_user * p1x

    sse_user = float(((sub[y_col] - (m_user * sub[x_col] + b_user)) ** 2).sum())
    sse_ols = float(((sub[y_col] - ols.predict(sub[[x_col]])) ** 2).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("相關係數 r", f"{r_val:.3f}")
    m2.metric("你的線 SSE", f"{sse_user:,.0f}")
    m3.metric(
        "OLS 最佳解 SSE", f"{sse_ols:,.0f}",
        delta=f"{sse_ols - sse_user:,.0f}", delta_color="inverse",
    )

    xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 60)
    fig = px.scatter(
        sub, x=x_col, y=y_col, opacity=0.35,
        labels={x_col: f"第 {exam} 次模考總分", y_col: "統測總分"},
    )
    fig.add_scatter(
        x=xs, y=m_user * xs + b_user,
        mode="lines", name="你的線", line=dict(dash="dash", width=3),
    )
    fig.add_scatter(
        x=xs, y=m_ols * xs + b_ols,
        mode="lines", name="OLS 最佳解", line=dict(width=3),
    )
    fig.add_scatter(
        x=[p1x, p2x], y=[p1y, p2y],
        mode="markers", name="你拉的兩點",
        marker=dict(size=14, symbol="x", color="red"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"OLS 公式：統測總分 ≈ **{m_ols:.2f}** × 模{exam}總分 + **{b_ols:.1f}**　｜　"
        f"樣本 n = {len(sub):,}"
    )
    st.info(
        "💡 小挑戰：OLS 的 SSE 是「所有可能的直線中最小的」——"
        "不管你怎麼拉，你的 SSE 永遠 ≥ OLS 的 SSE。這就是「最小平方法」名字的由來。"
    )

st.divider()

# ────────────────────────────────────────────────────────────────
# Section 3 你考幾分可以上哪裡？
# ────────────────────────────────────────────────────────────────
st.header("③ 你考幾分可以上哪裡？（落點查詢）")
st.caption("數學概念：加權平均數、區間、排序、中位數")
st.caption("資料來源：112/113/114 科技校院四技二專聯合登記分發錄取統計（09 商業與管理群）")

th = load_thresholds()

st.info(
    "💡 **為什麼要分科輸入？**　不同學校的錄取門檻用的公式不同——"
    "有些學校英文加權 ×2、有些專二加權 ×3，所以「950 分」和「560 分」不能直接比高低，"
    "要先**把你的各科分數乘上那間學校的權重**，才能和它的門檻比較。"
)

# 各科分數輸入
st.subheader("① 輸入你的統測各科分數（0–100）")

# 「用自己科的中位數填入」快速鍵
SUBJ_COLS_MAP = {"國文": "統測_國文分數", "英文": "統測_英文分數",
                 "數學": "統測_數學B分數", "專一": "統測_專一分數", "專二": "統測_專二分數"}
dept_subj_medians = {
    dept: {s: round(wide[wide["科別"] == dept][col].median(), 1)
           for s, col in SUBJ_COLS_MAP.items()}
    for dept in DEPT_ORDER
}

dept_options = ["（自行輸入）"] + DEPT_ORDER
qc1, qc2 = st.columns([2, 1])
pick_dept = qc1.selectbox("用自己科的中位數填入？", dept_options, key="q3_dept")
apply_btn = qc2.button("套用科別中位數", key="q3_apply")

for s in ["國文", "英文", "數學", "專一", "專二"]:
    if f"s3_{s}" not in st.session_state:
        st.session_state[f"s3_{s}"] = 68

if apply_btn and pick_dept != "（自行輸入）":
    for s, v in dept_subj_medians[pick_dept].items():
        st.session_state[f"s3_{s}"] = int(v)

sc1, sc2, sc3, sc4, sc5 = st.columns(5)
s_國文 = sc1.number_input("國文", 0, 100, key="s3_國文")
s_英文 = sc2.number_input("英文", 0, 100, key="s3_英文")
s_數學 = sc3.number_input("數學", 0, 100, key="s3_數學")
s_專一 = sc4.number_input("專一", 0, 100, key="s3_專一")
s_專二 = sc5.number_input("專二", 0, 100, key="s3_專二")

user_scores = {"國文": s_國文, "英文": s_英文, "數學": s_數學, "專一": s_專一, "專二": s_專二}
raw_total = sum(user_scores.values())
st.caption(f"原始總分（等權）：{raw_total} 分　|　最高 500 分")

# 篩選條件
st.subheader("② 選擇查詢條件")
fc1, fc2 = st.columns(2)
cat = fc1.radio("科系分類", ["商管類", "資訊類", "語文類", "設計類"], horizontal=True)
year_q = fc2.selectbox("查哪一年的門檻", [114, 113, 112])

# 計算每校加權分數並比對門檻
th_sub = th[(th["年度"] == year_q) & (th["科系分類"] == cat)].copy()

def row_weighted(row):
    try:
        w = parse_weights(row["各科目加權"])
        return round(calc_weighted(user_scores, w), 2)
    except Exception:
        return None

th_sub["你的加權分"] = th_sub.apply(row_weighted, axis=1)
th_sub["分差"] = (th_sub["錄取總分數"] - th_sub["你的加權分"]).round(1)
th_sub["最高可能分"] = th_sub["各科目加權"].apply(
    lambda f: round(sum(parse_weights(f).values()) * 100, 0) if pd.notna(f) else None
)

hit = th_sub[th_sub["分差"].abs() <= 30].dropna(subset=["你的加權分"])

st.write(f"**共找到 {len(hit)} 個志願**（門檻 ± 30 分內），按分差絕對值排序：")
st.dataframe(
    hit.sort_values("分差", key=lambda s: s.abs())[
        ["學校名稱", "系科組學程名稱", "學校類型",
         "各科目加權", "最高可能分", "你的加權分", "錄取總分數", "分差"]
    ].reset_index(drop=True),
    width="stretch",
    hide_index=False,
)
st.caption("「最高可能分」= 各科滿分 100 × 該校加權總和，方便和你的加權分對比比例。")

st.divider()

# ────────────────────────────────────────────────────────────────
# Section 4 熱門學校／科系排行
# ────────────────────────────────────────────────────────────────
st.header("④ 熱門學校／科系排行")
st.caption("數學概念：排序、Top-K、盒鬚圖（分位數）")

tab1, tab2, tab3 = st.tabs(["學校 Top 10", "科系分類分佈", "錄取分數盒鬚圖"])

with tab1:
    year_t = st.selectbox("年度", [114, 113, 112], key="t1_year")
    type_filter = st.multiselect(
        "學校類型", ["國立科大", "國立大學", "私立科大", "私立大學"],
        default=["國立科大", "國立大學", "私立科大", "私立大學"], key="t1_type",
    )
    th_sub = th[(th["年度"] == year_t) & (th["學校類型"].isin(type_filter))]
    top = (
        th_sub.groupby(["學校名稱", "學校類型"], as_index=False)
        .agg(志願數=("志願代碼", "count"), 平均錄取分=("錄取總分數", "mean"))
        .sort_values("志願數", ascending=False).head(10)
    )
    top["平均錄取分"] = top["平均錄取分"].round(1)
    fig = px.bar(
        top, x="志願數", y="學校名稱", color="學校類型", orientation="h",
        hover_data=["平均錄取分"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

with tab2:
    year_t = st.selectbox("年度", [114, 113, 112], key="t2_year")
    cat_count = (
        th[th["年度"] == year_t]["科系分類"]
        .value_counts().reset_index()
    )
    cat_count.columns = ["科系分類", "志願數"]
    fig = px.bar(cat_count, x="科系分類", y="志願數", color="科系分類", text="志願數")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

with tab3:
    year_t = st.selectbox("年度", [114, 113, 112], key="t3_year")
    box_sub = th[th["年度"] == year_t].dropna(subset=["錄取總分數"])
    fig = px.box(
        box_sub, x="學校類型", y="錄取總分數", color="學校類型",
        points="outliers", category_orders={
            "學校類型": ["國立科大", "國立大學", "私立科大", "私立大學", "其他"]
        },
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "盒鬚圖解讀：盒子 = Q1~Q3（中間 50% 資料）、中間線 = 中位數、"
        "鬚線 = 1.5 倍 IQR 範圍、離群點獨立顯示。"
    )

st.divider()

# ────────────────────────────────────────────────────────────────
# Section 5 幕後程式
# ────────────────────────────────────────────────────────────────
st.header("⑤ 幕後程式：Python 怎麼做到的？")
st.caption("這整個網頁就是一支 Python 程式（約 200 行）。以下是每個 section 的關鍵 5 行：")

with st.expander("Section 1 — 科別錄取地圖（groupby + 長條圖）"):
    st.code("""
DEPT_MAP = {"1": "商經科", "2": "國貿科", "4": "資處科"}
df["科別"] = df["班級_標準"].str[0].map(DEPT_MAP)

sub_adm = adm[adm["科別"] == sel_dept]
counts = sub_adm["學校類型"].value_counts().reset_index()
counts.columns = ["學校類型", "人次"]
fig = px.bar(counts, x="學校類型", y="人次", color="學校類型")
st.plotly_chart(fig)
""", language="python")

with st.expander("Section 2 — 相關係數 + OLS 最小平方迴歸"):
    st.code("""
sub = wide[[x_col, y_col]].dropna()
r_val = sub.corr().iloc[0, 1]                         # 相關係數

from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(sub[[x_col]], sub[y_col])
m_ols, b_ols = ols.coef_[0], ols.intercept_            # y = m·x + b

sse = ((sub[y_col] - ols.predict(sub[[x_col]])) ** 2).sum()  # SSE
""", language="python")

with st.expander("Section 3 — 落點查詢（條件篩選 + 分差排序）"):
    st.code("""
hit = th[
    (th["年度"] == year_q)
    & (th["科系分類"] == cat)
    & (th["錄取總分數"].between(score - 30, score + 30))
].copy()
hit["分差"] = hit["錄取總分數"] - score
hit.sort_values("分差", key=lambda s: s.abs())
""", language="python")

with st.expander("Section 4 — 分位數與盒鬚圖"):
    st.code("""
box_sub = th[th["年度"] == year_t]
fig = px.box(box_sub, x="學校類型", y="錄取總分數", color="學校類型")
# 盒子 = Q1~Q3，中線 = 中位數 = 50% 分位數
# 鬚線 = 1.5 * IQR，超出就是離群點
""", language="python")

st.success(
    "🎯 想自己改看看？打開 `site/app.py`，改 `sel_class` 的預設值、"
    "或是把 `px.bar` 換成 `px.pie`，存檔後網頁會自動重新整理。"
)
