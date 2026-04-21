"""
互動簡報模式：資訊 × 數學共同課
啟動：streamlit run site/slides.py
"""
from __future__ import annotations

import re
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_wide

st.set_page_config(
    page_title="統測資料課",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }
/* 保留 sidebar 收合按鈕 */
[data-testid="collapsedControl"] { visibility: visible !important; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 6rem !important;
    max-width: 100% !important;
    padding-left: 4vw !important;
    padding-right: 4vw !important;
}
/* 底部固定導覽列 */
.snav {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: rgba(14,17,23,0.97);
    border-top: 1px solid #2d2d2d;
    padding: 12px 40px;
    z-index: 9999;
    display: flex; justify-content: space-between; align-items: center;
}
.snav-btn {
    background: #ff4b4b; color: white;
    padding: 10px 32px; font-size: 18px; font-weight: bold;
    border-radius: 8px; border: none; cursor: pointer; letter-spacing: 1px;
}
.snav-btn.prev { background: #444; }
.snav-btn:disabled { background: #222; color: #555; cursor: default; }
.snav-pg { color: #666; font-size: 16px; font-family: monospace; }
h1.st { font-size: 58px !important; line-height: 1.25 !important; }
h2.st { font-size: 42px !important; line-height: 1.3  !important; }
.big  { font-size: 24px; line-height: 1.9; }
.center { text-align: center; }
.full {
    min-height: calc(100vh - 140px);
    display: flex; flex-direction: column; justify-content: center;
}
.tag {
    display: inline-block; background: #ff4b4b; color: white;
    padding: 5px 20px; border-radius: 20px; font-size: 18px; margin-bottom: 16px;
}
.tag.phase { background: #1a6bb5; }
</style>
""", unsafe_allow_html=True)

# ── 資料 ──────────────────────────────────────────────────────────
DEPT_MAP   = {"1": "商經科", "2": "國貿科", "4": "資處科"}
DEPT_ORDER = ["商經科", "國貿科", "資處科"]
DEPT_COLOR = px.colors.qualitative.Bold[:3]
SUBJ = {
    "國文": "統測_國文分數", "英文": "統測_英文分數",
    "數學": "統測_數學B分數",
    "專一": "統測_專一分數", "專二": "統測_專二分數",
}

@st.cache_data
def get_wide():
    w = load_wide()
    w["科別"] = w["班級_標準"].astype(str).str[0].map(DEPT_MAP)
    return w[w["科別"].notna()].reset_index(drop=True)

wide = get_wide()

# ── 投影片目錄（sidebar 顯示用）─────────────────────────────────────
SLIDES_META = [
    ("🏠", "標題頁",            ""),
    ("①", "第一階段說明",        "phase"),
    ("①", "原始資料震撼",        ""),
    ("①", "猜謎：哪科最低",      ""),
    ("①", "揭曉：各科平均",      ""),
    ("①", "陷阱題投票",          ""),
    ("①", "直方圖揭曉",          ""),
    ("②", "第二階段說明",        "phase"),
    ("②", "正相關 vs 負相關",      ""),
    ("②", "相關係數 r 怎麼算",    ""),
    ("②", "分組討論計時",        ""),
    ("③", "第三階段說明",        "phase"),
    ("③", "折線 + 長條圖",       ""),
    ("③", "箱型圖",              ""),
    ("③", "Q1/Q2/Q3/IQR",       ""),
    ("③", "EDA 總結",            ""),
    ("④", "第四階段說明",        "phase"),
    ("④", "迴歸直線怎麼算？",   ""),
    ("④", "提問：能預測嗎？",    ""),
    ("④", "相關係數 r 的變化",   ""),
    ("④", "預測實機展示",        ""),
    ("④", "y = ax + b 解說",    ""),
    ("④", "你來畫迴歸線",        ""),
    ("④", "殘差與 SSE 視覺化",   ""),
    ("④", "模考→統測→落點",      ""),
    ("④", "推甄 vs 分發",        ""),
    ("④", "總結：打破預測",      ""),
]
N = len(SLIDES_META)

# ── 狀態初始化（同時支援 query param 跳轉）─────────────────────────
_params = st.query_params
if "slide" in _params:
    try:
        _p = int(_params["slide"])
        st.session_state.slide = _p
        st.query_params.clear()          # 讀完就清，避免刷新後卡住
    except Exception:
        pass
if "slide" not in st.session_state:
    st.session_state.slide = 0
slide = max(0, min(N - 1, st.session_state.slide))

# ── phase 顏色對照 ─────────────────────────────────────────────────
_PHASE_BG = {"🏠": "#444", "①": "#7b2020", "②": "#1a3d6b",
             "③": "#1a5c3a", "④": "#4a2070"}
_PHASE_FG = {"🏠": "#aaa", "①": "#ff8080", "②": "#6baeff",
             "③": "#6bffaa", "④": "#c080ff"}

# ── 縮圖 HTML 產生器 ───────────────────────────────────────────────
def _build_thumb_html(current: int) -> str:
    CONTENT_ICON = [
        "🎯","📣","🗂️","🤔","✅","⚠️","📊",
        "📣","🔄","🧮","⏱️",
        "📣","📈","📦","📐","🧐",
        "📣","📐","🤔","📈","🔮","➗","✏️","📏","🎯","⚖️","🏆",
    ]
    cards = []
    for i, (icon, title, kind) in enumerate(SLIDES_META):
        bg_dim  = _PHASE_BG.get(icon, "#333")
        fg      = _PHASE_FG.get(icon, "#ccc")
        active  = "border:2px solid #fff;" if i == current else "border:1px solid #555;"
        opacity = "opacity:1;" if i == current else "opacity:0.75;"
        short   = title[:7] + ("…" if len(title) > 7 else "")
        cicon   = CONTENT_ICON[i] if i < len(CONTENT_ICON) else icon
        cards.append(f"""
<div class="card" style="background:{bg_dim};{active}{opacity}"
     onclick="window.parent.postMessage({{type:'goto_slide',n:{i}}},'*')"
     title="{i+1}. {title}">
  <div class="ci">{cicon}</div>
  <div class="num" style="color:{fg}">{i+1}</div>
  <div class="ttl">{short}</div>
</div>""")

    return f"""
<style>
  body {{ margin:0; background:#0e1117; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; padding:4px; }}
  .card {{ border-radius:7px; padding:8px 4px; text-align:center;
           cursor:pointer; transition:opacity .15s, border .15s; }}
  .card:hover {{ opacity:1 !important; border:2px solid #aaa !important; }}
  .ci  {{ font-size:20px; line-height:1; }}
  .num {{ font-size:13px; font-weight:bold; margin-top:3px; }}
  .ttl {{ font-size:9px; color:#aaa; margin-top:2px; line-height:1.3; }}
</style>
<div class="grid">{''.join(cards)}</div>"""

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    view = st.radio("", ["📋 目錄", "🖼 縮圖"],
                    horizontal=True, key="sb_view",
                    label_visibility="collapsed")

    if view == "🖼 縮圖":
        # 計算需要多少高度：每 2 張一行，每行約 90px
        rows = (N + 1) // 2
        h    = rows * 92 + 16
        components.html(_build_thumb_html(slide), height=h, scrolling=True)
    else:
        # 目錄模式
        st.markdown("""
<style>
section[data-testid="stSidebar"] .stButton button {
    padding: 4px 8px !important; font-size: 12px !important;
    min-height: 30px !important;
}
</style>""", unsafe_allow_html=True)
        for i, (icon, title, kind) in enumerate(SLIDES_META):
            if kind == "phase":
                st.markdown(
                    f"<div style='margin:8px 0 2px;font-size:11px;"
                    f"color:#888;font-weight:bold;letter-spacing:1px'>"
                    f"{icon} {title}</div>", unsafe_allow_html=True)
                if st.button("▶ 跳至此頁", key=f"sb_{i}",
                             use_container_width=True,
                             type="primary" if i == slide else "secondary"):
                    st.session_state.slide = i; st.rerun()
            else:
                label = f"{icon} {i+1}. {title}"
                if st.button(label, key=f"sb_{i}", use_container_width=True,
                             type="primary" if i == slide else "secondary"):
                    st.session_state.slide = i; st.rerun()

# ── 底部導覽（session_state 驅動，不開新頁）──────────────────────
def nav_bar(s, n):
    icon, title, _ = SLIDES_META[s]
    st.markdown(f"""
<div class="snav">
  <span id="snav-left"></span>
  <span class="snav-pg">{s+1} / {n}　{icon} {title}</span>
  <span id="snav-right"></span>
</div>
<style>
/* 讓最後一個 stHorizontalBlock 的按鈕貼近底部導覽視覺 */
div[data-testid="stHorizontalBlock"]:last-of-type button {{
    height: 48px; font-size: 18px; font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if s > 0:
            if st.button("← 上一頁", key="nav_prev", use_container_width=True):
                st.session_state.slide = s - 1; st.rerun()
    with col3:
        if s < n - 1:
            if st.button("下一頁 →", key="nav_next", use_container_width=True):
                st.session_state.slide = s + 1; st.rerun()

# 鍵盤左右鍵 + 縮圖 postMessage 跳頁
components.html("""
<script>
function clickByText(text) {
  const btns = window.parent.document.querySelectorAll('button');
  for (const b of btns) {
    if (b.innerText && b.innerText.includes(text) && !b.disabled) {
      b.click(); return;
    }
  }
}
window.parent.document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowRight' || e.key === ' ') clickByText('下一頁');
  else if (e.key === 'ArrowLeft')               clickByText('上一頁');
});

// 接收縮圖點擊的 postMessage，找 sidebar 第 n 個按鈕點擊
window.parent.addEventListener('message', function(e) {
  if (!e.data || e.data.type !== 'goto_slide') return;
  const n = e.data.n;
  const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
  if (!sidebar) return;
  const btns = sidebar.querySelectorAll('button');
  if (btns[n]) btns[n].click();
});
</script>
""", height=0)

# ═══════════════════════════════════════════════════════════════════
# 投影片內容
# ═══════════════════════════════════════════════════════════════════

def s0():  # 標題
    st.markdown("""
<div class="full center">
  <div style="font-size:24px;color:#ff4b4b;margin-bottom:10px">資訊 × 數學 共同教學</div>
  <h1 class="st">用真實資料<br>理解統計與預測</h1>
  <div class="big" style="color:#888;margin-top:32px">
    112–114 年三屆學長姐的統測成績<br>
    <span style="font-size:20px">商經科 · 國貿科 · 資處科</span>
  </div>
  <div style="margin-top:40px;font-size:18px;color:#555">→ 鍵盤右鍵 / Space 翻頁，或用左側目錄跳頁</div>
</div>""", unsafe_allow_html=True)


def s1():  # 第一階段說明
    st.markdown('<span class="tag phase">第一階段 · 約 25 分鐘</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="st">數據破冰與直覺挑戰</h1>', unsafe_allow_html=True)
    st.markdown("""
<div class="big"><br>
🎯 &nbsp;目標：引起動機，打破主觀直覺，建立數據意識<br><br>
<b>活動 1</b> &nbsp;&nbsp;統測「大魔王」猜謎<br>
<b>活動 2</b> &nbsp;&nbsp;陷阱題：平均數 vs. 中位數
</div>""", unsafe_allow_html=True)


def s2():  # 原始資料
    st.markdown('<span class="tag">活動 1 · 資訊老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">這就是「原始資料」</h2>', unsafe_allow_html=True)
    st.caption(f"共 {len(wide):,} 位學長姐 · {wide.shape[1]} 個欄位 · 每一列就是一個人")
    st.dataframe(wide.drop(columns=["科別"]), height=540, width="stretch")
    st.markdown("""<div class="big" style="margin-top:16px">
光用眼睛看，你能找出「哪一科的平均分數最低」嗎？
</div>""", unsafe_allow_html=True)


def s3():  # 投票
    st.markdown('<span class="tag">活動 1 · 即時投票</span>', unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center;padding-top:4vh">
<h1 class="st">你覺得統測哪一科<br>全班平均最低？</h1>
</div>""", unsafe_allow_html=True)

    if "v3" not in st.session_state:
        st.session_state.v3 = {s: 0 for s in SUBJ}
    cols = st.columns(5)
    for (s, _), col in zip(SUBJ.items(), cols):
        cnt = st.session_state.v3[s]
        if col.button(f"**{s}**\n\n{cnt} 票", key=f"v3_{s}", use_container_width=True):
            st.session_state.v3[s] += 1; st.rerun()

    st.markdown("<div style='text-align:center;color:#555;font-size:18px;margin-top:12px'>"
                "老師點按加票（或接 Slido / Kahoot）</div>", unsafe_allow_html=True)

    if st.button("🔍 揭曉答案", key="reveal3"):
        st.session_state.revealed3 = True
    if st.session_state.get("revealed3"):
        avgs = {s: wide[col].mean() for s, col in SUBJ.items()}
        lowest = min(avgs, key=avgs.get)
        st.success(f"✅  答案是：**{lowest}**（平均 {avgs[lowest]:.1f} 分）")
        for s, col in zip(SUBJ.keys(), st.columns(5)):
            col.metric(s, f"{avgs[s]:.1f}")


def s4():  # 揭曉長條圖
    st.markdown('<span class="tag">活動 1 · 揭曉</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">兩行程式碼秒出答案</h2>', unsafe_allow_html=True)
    avgs = {s: round(wide[col].mean(), 1) for s, col in SUBJ.items()}
    c_code, c_chart = st.columns([1, 2])
    with c_code:
        st.code("df[ subj_cols ].mean()\n\n# 就這樣", language="python")
        st.dataframe(pd.DataFrame(avgs.items(), columns=["科目","平均分數"])
                       .sort_values("平均分數"), hide_index=True, height=260)
    with c_chart:
        df = pd.DataFrame(avgs.items(), columns=["科目","平均分數"]).sort_values("平均分數")
        fig = px.bar(df, x="科目", y="平均分數", color="科目", text="平均分數",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_traces(textposition="outside", textfont_size=22)
        fig.update_layout(font_size=18, height=500, showlegend=False, yaxis_range=[0, 115])
        st.plotly_chart(fig, width="stretch")


def s5():  # 陷阱題
    avgs = {s: wide[col].mean() for s, col in SUBJ.items()}
    lowest_s, lowest_v = min(avgs, key=avgs.get), round(min(avgs.values()), 1)
    st.markdown('<span class="tag">活動 2 · 數學老師</span>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="full center">
<h1 class="st">{lowest_s} 的平均是 {lowest_v} 分<br><br>
那「中位數」會比平均數<br>
<span style="color:#ff4b4b">高</span> 還是 <span style="color:#4b9eff">低</span>？
</h1></div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    hi = st.session_state.get("v5_hi", 0)
    lo = st.session_state.get("v5_lo", 0)
    if c1.button(f"🙋  比較**高**　　{hi} 人", key="v5h", use_container_width=True):
        st.session_state.v5_hi = hi + 1; st.rerun()
    if c2.button(f"🙋  比較**低**　　{lo} 人", key="v5l", use_container_width=True):
        st.session_state.v5_lo = lo + 1; st.rerun()


def s6():  # 直方圖揭曉
    avgs = {s: wide[col].mean() for s, col in SUBJ.items()}
    lowest_s = min(avgs, key=avgs.get)
    col_key  = SUBJ[lowest_s]
    data     = wide[col_key].dropna()
    mean_v, med_v = data.mean(), data.median()
    st.markdown('<span class="tag">活動 2 · 揭曉</span>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="st">{lowest_s} 成績分布直方圖</h2>', unsafe_allow_html=True)
    c_chart, c_exp = st.columns([3, 1])
    with c_chart:
        fig = px.histogram(data, nbins=35, labels={"value": f"{lowest_s} 分數", "count": "人數"})
        fig.add_vline(x=mean_v, line_dash="dash", line_color="#ff4b4b", line_width=3,
                      annotation_text=f"平均 {mean_v:.1f}", annotation_font_size=18)
        fig.add_vline(x=med_v, line_color="#4bff91", line_width=3,
                      annotation_text=f"中位數 {med_v:.1f}", annotation_font_size=18,
                      annotation_position="bottom right")
        fig.update_layout(font_size=18, height=520)
        st.plotly_chart(fig, width="stretch")
    diff = med_v - mean_v
    direction = "左偏（負偏）" if diff > 0 else "右偏（正偏）"
    with c_exp:
        st.markdown(f"""
<div class="big">
<b>平均數</b><br>{mean_v:.1f}<br><br>
<b>中位數</b><br>{med_v:.1f}<br><br>
<b>差距</b><br>{abs(diff):.1f} 分<br><br>
<span style="color:#ff4b4b">{direction}</span><br><br>
少數極低分<br>把平均往下拖
</div>""", unsafe_allow_html=True)


def s7():  # 第二階段
    st.markdown('<span class="tag phase">第二階段 · 約 20 分鐘</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="st">迷思破解與深度探索</h1>', unsafe_allow_html=True)
    st.markdown("""
<div class="big"><br>
🎯 &nbsp;目標：從數據中找關聯，培養問對問題的能力<br><br>
<b>活動 3</b> &nbsp;&nbsp;「國文好、數學差」的文組傳說存在嗎？<br>
<b>活動 4</b> &nbsp;&nbsp;學生分組發想 5 分鐘
</div>""", unsafe_allow_html=True)


def s8a():  # 教學：正相關 / 負相關 / 無相關
    st.markdown('<span class="tag">活動 3 · 數學老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">什麼是相關性？</h2>', unsafe_allow_html=True)

    rng = np.random.default_rng(42)
    n = 80
    x = np.linspace(10, 90, n)
    examples = [
        ("正相關 📈", x * 0.8 + rng.normal(0, 10, n),    "#4bff91",
         "x 大 → y 也大\nr 接近 **+1**\n例：練習時數 vs 成績"),
        ("負相關 📉", -x * 0.8 + 100 + rng.normal(0, 10, n), "#ff6b6b",
         "x 大 → y 反而小\nr 接近 **−1**\n例：缺課次數 vs 成績"),
        ("無相關 ➡", rng.normal(50, 18, n),               "#888888",
         "x 和 y 沒有線性關係\nr 接近 **0**\n例：鞋號 vs 數學成績"),
    ]

    cols = st.columns(3)
    for col, (title, y, color, desc) in zip(cols, examples):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                                 marker=dict(color=color, size=6, opacity=0.65)))
        if "無相關" not in title:
            m, b = np.polyfit(x, y, 1)
            fig.add_trace(go.Scatter(x=[x.min(), x.max()],
                                     y=[m*x.min()+b, m*x.max()+b],
                                     mode="lines", line=dict(color=color, width=3)))
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10), height=260,
            xaxis=dict(showticklabels=False, title="x"),
            yaxis=dict(showticklabels=False, title="y"),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        col.plotly_chart(fig, use_container_width=True)
        col.markdown(f"**{title}**")
        col.markdown(desc)

    st.markdown("---")
    st.markdown("""
<div class="big">
📐 &nbsp;<b>相關係數 r</b>（Pearson）是介於 <b>−1 到 +1</b> 的數字<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;r = +1　完美正相關（所有點在一條斜向上直線）<br>
&nbsp;&nbsp;&nbsp;&nbsp;r =  0　無線性相關<br>
&nbsp;&nbsp;&nbsp;&nbsp;r = −1　完美負相關（所有點在一條斜向下直線）<br><br>
❓ 那「國文高分的人，數學分數」是正相關、負相關、還是無相關？
</div>""", unsafe_allow_html=True)


def s8b():  # 教學：r 公式 + 計算 + 揭曉
    st.markdown('<span class="tag">活動 3 · 數學老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">相關係數 r 怎麼算？</h2>', unsafe_allow_html=True)

    sub = wide[["統測_國文分數", "統測_數學B分數", "科別"]].dropna().reset_index(drop=True)
    x_bar = sub["統測_國文分數"].mean()
    y_bar = sub["統測_數學B分數"].mean()
    sx    = sub["統測_國文分數"].std()
    sy    = sub["統測_數學B分數"].std()
    r     = sub[["統測_國文分數","統測_數學B分數"]].corr().iloc[0, 1]
    n     = len(sub)

    tab_f, tab_calc, tab_reveal = st.tabs(["📐 公式", "🔢 代入計算（國文 vs 數學）", "✅ 揭曉趨勢線"])

    with tab_f:
        st.markdown("#### 相關係數公式（Pearson）")
        st.latex(r"r = \frac{\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{(n-1)\cdot s_x \cdot s_y}")
        st.markdown("""
<div class="big">
📌 &nbsp;記憶方式<br>
&nbsp;&nbsp;&nbsp;<b>分子</b> = 「x 偏差 × y 偏差」的總和<br>
&nbsp;&nbsp;&nbsp;<b>分母</b> = (樣本數−1) × x 的標準差 × y 的標準差<br><br>
💡 也可以這樣理解：把每個點先各自「標準化」再乘起來<br>
</div>""", unsafe_allow_html=True)
        st.latex(r"r = \frac{1}{n-1}\sum_{i=1}^{n}\frac{x_i-\bar{x}}{s_x}\cdot\frac{y_i-\bar{y}}{s_y}")
        st.info("每個 (xi−x̄)/sx 就是 **z-score（標準分數）**；r 就是兩組 z-score 乘積的平均！")

    with tab_calc:
        st.markdown(f"""
<div class="big">
<b>Step 1</b>：算平均數<br>
&nbsp;&nbsp;x̄（國文）= {x_bar:.1f}　　ȳ（數學）= {y_bar:.1f}<br><br>
<b>Step 2</b>：算標準差<br>
&nbsp;&nbsp;s_x = {sx:.2f}　　s_y = {sy:.2f}<br><br>
<b>Step 3</b>：把每筆資料的 (xi−x̄)(yi−ȳ) 算出來，再全部加總<br><br>
<b>Step 4</b>：除以 (n−1)·s_x·s_y<br>
&nbsp;&nbsp;r = <span style="font-size:36px;color:#ff4b4b;font-weight:bold">{r:.3f}</span>　（n = {n:,}）<br><br>
{'📈 正相關：國文越高，數學也傾向較高' if r > 0 else '📉 負相關'}
</div>""", unsafe_allow_html=True)
        sample = sub.sample(6, random_state=7).copy()
        sample["xi−x̄"] = (sample["統測_國文分數"] - x_bar).round(1)
        sample["yi−ȳ"]  = (sample["統測_數學B分數"] - y_bar).round(1)
        sample["(xi−x̄)(yi−ȳ)"] = (sample["xi−x̄"] * sample["yi−ȳ"]).round(1)
        sample["(xi−x̄)/sx"]    = (sample["xi−x̄"] / sx).round(3)
        sample["(yi−ȳ)/sy"]    = (sample["yi−ȳ"] / sy).round(3)
        sample = sample.rename(columns={"統測_國文分數":"國文","統測_數學B分數":"數學"})
        st.dataframe(sample[["國文","數學","xi−x̄","yi−ȳ","(xi−x̄)(yi−ȳ)","(xi−x̄)/sx","(yi−ȳ)/sy"]],
                     hide_index=True)
        st.caption("每一列就是一位學長姐；把最後一欄全部加起來再除以 (n−1) 就是 r。")

    with tab_reveal:
        _ols = LinearRegression().fit(sub[["統測_國文分數"]], sub["統測_數學B分數"])
        xs   = np.linspace(sub["統測_國文分數"].min(), sub["統測_國文分數"].max(), 60)
        c_chart, c_text = st.columns([3, 1])
        with c_chart:
            fig = px.scatter(sub, x="統測_國文分數", y="統測_數學B分數",
                             color="科別", opacity=0.35,
                             labels={"統測_國文分數":"國文分數","統測_數學B分數":"數學分數"},
                             color_discrete_sequence=DEPT_COLOR)
            fig.add_scatter(x=xs, y=_ols.coef_[0]*xs+_ols.intercept_,
                            mode="lines", name="OLS 趨勢線",
                            line=dict(color="#ffcc00", width=4))
            fig.update_layout(font_size=18, height=480)
            st.plotly_chart(fig, width="stretch")
        with c_text:
            trend = "正相關 📈" if r > 0.1 else ("負相關 📉" if r < -0.1 else "幾乎無相關")
            st.markdown(f"""
<div class="big">
<b>相關係數 r</b><br>
<span style="font-size:56px;color:#ff4b4b;font-weight:bold">{r:.2f}</span><br><br>
{trend}<br><br>
打破「文武不能雙全」迷思！<br><br>
⚠️ 相關性<br>≠ 因果關係
</div>""", unsafe_allow_html=True)


def s9():  # 分組討論
    st.markdown('<span class="tag">活動 4 · 學生主導</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="st">你們還能挖出什麼秘密？</h1>', unsafe_allow_html=True)
    c_prompts, c_timer = st.columns([3, 1])
    with c_prompts:
        st.markdown("""
<div class="big">
小組討論 5 分鐘，選一個感興趣的問題：<br><br>
💡 &nbsp;專一和專二，哪科對總分影響更大？<br><br>
💡 &nbsp;模考成績有隨著模1→模5 進步嗎？<br><br>
💡 &nbsp;哪個科別在哪一科特別突出？<br><br>
💡 &nbsp;你自己想到的問題……
</div>""", unsafe_allow_html=True)
    with c_timer:
        if "ts" not in st.session_state:
            st.session_state.ts = None
        if st.button("▶ 開始計時", key="start_t", use_container_width=True):
            st.session_state.ts = time.time(); st.rerun()
        if st.session_state.ts:
            remaining = max(0, 300 - int(time.time() - st.session_state.ts))
            m, s_rem = divmod(remaining, 60)
            color = ("#ff4b4b" if remaining < 60 else "#ffaa00" if remaining < 120 else "#4bff91")
            st.markdown(f"""
<div style="text-align:center">
<div style="font-size:100px;font-weight:bold;color:{color};line-height:1">{m:02d}:{s_rem:02d}</div>
<div style="font-size:20px;color:#888;margin-top:6px">剩餘時間</div>
</div>""", unsafe_allow_html=True)
            if remaining > 0:
                time.sleep(1); st.rerun()
            else:
                st.error("⏰ 時間到！")
                if st.button("重置", key="rst_t"):
                    st.session_state.ts = None; st.rerun()


def s10():  # 第三階段
    st.markdown('<span class="tag phase">第三階段 · 約 25 分鐘</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="st">資料視覺化（EDA）實作</h1>', unsafe_allow_html=True)
    st.markdown("""
<div class="big"><br>
🎯 &nbsp;目標：將抽象數字轉化為視覺圖表，精準解讀背後的數學意義<br><br>
<b>活動 5</b> &nbsp;&nbsp;折線圖 + 長條圖 + 箱型圖<br>
<b>活動 6</b> &nbsp;&nbsp;Q₁ / Q₂ / Q₃ / IQR 的數學意義<br>
<b>活動 7</b> &nbsp;&nbsp;EDA 總結
</div>""", unsafe_allow_html=True)


def s11():  # 折線 + 長條
    st.markdown('<span class="tag">活動 5 · 資訊老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">三種圖表・三種角度</h2>', unsafe_allow_html=True)
    tab_line, tab_bar = st.tabs(["📈 折線圖：模考成績有進步嗎？","📊 長條圖：各科別 × 各科目"])
    with tab_line:
        dept_sel = st.multiselect("科別", DEPT_ORDER, default=DEPT_ORDER, key="s11d")
        rows = [{"科別":d,"模考":f"模{i}",
                 "平均分數":round(wide[wide["科別"]==d][f"模{i}_總分數"].mean(),1)}
                for d in dept_sel for i in range(1,6)
                if pd.notna(wide[wide["科別"]==d][f"模{i}_總分數"].mean())]
        fig = px.line(pd.DataFrame(rows), x="模考", y="平均分數", color="科別",
                      markers=True, color_discrete_sequence=DEPT_COLOR)
        fig.update_traces(line_width=4, marker_size=14)
        fig.update_layout(font_size=18, height=460)
        st.plotly_chart(fig, width="stretch")
    with tab_bar:
        rows2 = [{"科別":d,"科目":s,"平均分數":round(wide[wide["科別"]==d][col].mean(),1)}
                 for s,col in SUBJ.items() for d in DEPT_ORDER
                 if pd.notna(wide[wide["科別"]==d][col].mean())]
        fig = px.bar(pd.DataFrame(rows2), x="科目", y="平均分數",
                     color="科別", barmode="group", text="平均分數",
                     color_discrete_sequence=DEPT_COLOR)
        fig.update_traces(textposition="outside", textfont_size=14)
        fig.update_layout(font_size=18, height=460, yaxis_range=[0,115])
        st.plotly_chart(fig, width="stretch")


def s12():  # 箱型圖
    st.markdown('<span class="tag">活動 5 · 重頭戲</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">箱型圖：一張圖看完所有分布</h2>', unsafe_allow_html=True)
    subj_sel = st.multiselect("選科目", list(SUBJ.keys()), default=list(SUBJ.keys()), key="s12s")
    rows = [{"科別":r["科別"],"科目":s,"分數":r[col]}
            for s in subj_sel for col in [SUBJ[s]]
            for _,r in wide[["科別",col]].dropna().iterrows()]
    fig = px.box(pd.DataFrame(rows), x="科目", y="分數", color="科別",
                 points="outliers", color_discrete_sequence=DEPT_COLOR,
                 category_orders={"科目":list(SUBJ.keys()),"科別":DEPT_ORDER})
    fig.update_layout(font_size=18, height=560)
    st.plotly_chart(fig, width="stretch")


def s13():  # Q1/Q2/Q3
    st.markdown('<span class="tag">活動 6 · 數學老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">箱型圖裡的數學</h2>', unsafe_allow_html=True)
    subj = st.selectbox("選一科來解說", list(SUBJ.keys()), key="s13s")
    data = wide[SUBJ[subj]].dropna()
    q1, q2, q3 = data.quantile([.25, .5, .75])
    iqr = q3 - q1
    n_out = int(((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum())
    c_chart, c_stats = st.columns([3, 1])
    with c_chart:
        fig = px.box(wide, y=SUBJ[subj], points="outliers",
                     labels={SUBJ[subj]: f"{subj} 分數"},
                     color_discrete_sequence=["#ff4b4b"])
        for val, label, pos in [
            (q1, f"Q₁ = {q1:.1f}（25%）",  "top right"),
            (q2, f"Q₂ = {q2:.1f}（中位數）","bottom right"),
            (q3, f"Q₃ = {q3:.1f}（75%）",  "top right"),
        ]:
            fig.add_hline(y=val, line_dash="dot", line_color="#aaa", line_width=2,
                          annotation_text=label, annotation_font_size=16,
                          annotation_position=pos)
        fig.update_layout(font_size=18, height=540)
        st.plotly_chart(fig, width="stretch")
    with c_stats:
        st.markdown(f"""
<div class="big">
<b>Q₁</b>（25%）<br>{q1:.1f}<br><br>
<b>Q₂</b> 中位數<br>{q2:.1f}<br><br>
<b>Q₃</b>（75%）<br>{q3:.1f}<br><br>
<b>IQR</b><br>{iqr:.1f}<br><br>
<span style="color:#ff4b4b">離群值</span><br>{n_out} 人
</div>""", unsafe_allow_html=True)


def s14():  # EDA 總結
    st.markdown("""
<div class="full center">
<span class="tag">活動 7 · 雙師總結</span><br><br>
<h1 class="st">「在下結論之前<br>先讓資料自己說話」</h1>
<div class="big" style="color:#888;margin-top:40px">
確認資料正確 ✓ &nbsp;&nbsp; 找出潛在趨勢 ✓ &nbsp;&nbsp; 提出值得深入的問題 ✓
</div>
</div>""", unsafe_allow_html=True)


def s15():  # 第四階段
    st.markdown('<span class="tag phase">第四階段 · 約 30 分鐘</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="st">走向未來——數據預測</h1>', unsafe_allow_html=True)
    st.markdown("""
<div class="big"><br>
🎯 &nbsp;目標：從「看見過去」走向「預測未來」<br><br>
<b>活動 8</b> &nbsp;&nbsp;第一次模考能決定統測成績嗎？（IT 老師）<br>
<b>活動 9</b> &nbsp;&nbsp;預測的底層邏輯：最小平方法（數學老師）
</div>""", unsafe_allow_html=True)


# ── 活動 8 前置：迴歸直線怎麼算？─────────────────────────────────────

def s15b():  # 迴歸直線教學（高職程度）
    st.markdown('<span class="tag">活動 8 前置 · 數學老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">迴歸直線怎麼算出來的？</h2>', unsafe_allow_html=True)

    exam = st.select_slider("選模考次數（以下用這次的真實資料示範）",
                             options=list(range(1, 6)), value=5, key="s15b_e")
    x_col = f"模{exam}_總分數"
    sub = wide[[x_col, "統測_總分數"]].dropna().reset_index(drop=True)

    x_bar = sub[x_col].mean()
    y_bar = sub["統測_總分數"].mean()
    sx    = sub[x_col].std()
    sy    = sub["統測_總分數"].std()
    r     = sub[[x_col, "統測_總分數"]].corr().iloc[0, 1]
    a     = r * sy / sx
    b     = y_bar - a * x_bar

    tab_step, tab_formula, tab_verify = st.tabs(
        ["📋 步驟說明", "📐 公式與代入", "✅ 驗證"])

    with tab_step:
        st.markdown(f"""
<div class="big">
我們要找一條 <b>y = ax + b</b>，使所有點的誤差²總和最小。<br><br>
<b>Step 1</b>　算出 x 和 y 的平均數<br>
&nbsp;&nbsp;&nbsp;&nbsp;x̄ = {x_bar:.1f}（模{exam}平均）　　ȳ = {y_bar:.1f}（統測平均）<br><br>
<b>Step 2</b>　算出標準差<br>
&nbsp;&nbsp;&nbsp;&nbsp;s_x = {sx:.2f}　　s_y = {sy:.2f}<br><br>
<b>Step 3</b>　算相關係數 r<br>
&nbsp;&nbsp;&nbsp;&nbsp;r = {r:.4f}<br><br>
<b>Step 4</b>　代入公式得到斜率 a 和截距 b<br>
&nbsp;&nbsp;&nbsp;&nbsp;a = r × (s_y ÷ s_x) = {r:.4f} × ({sy:.2f} ÷ {sx:.2f}) = <b>{a:.4f}</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;b = ȳ − a × x̄ = {y_bar:.1f} − {a:.4f} × {x_bar:.1f} = <b>{b:.1f}</b>
</div>""", unsafe_allow_html=True)

    with tab_formula:
        st.markdown("#### 斜率公式（兩種等價寫法）")
        st.latex(r"a = r \cdot \frac{s_y}{s_x}")
        st.latex(r"a = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}")
        st.markdown("#### 截距公式")
        st.latex(r"b = \bar{y} - a \cdot \bar{x}")
        st.info("💡 **記憶技巧**：直線一定過 (x̄, ȳ) 這個點——"
                "把 x = x̄ 代入 y = ax̄ + b，你就能理解截距公式從哪來的。")

    with tab_verify:
        st.markdown(f"#### 代入數字驗算")
        st.markdown(f"""
<div class="big">
迴歸方程式：y = <b>{a:.4f}</b> × x + <b>{b:.1f}</b><br><br>
驗算 1：把 x = x̄ = {x_bar:.1f} 代入<br>
&nbsp;&nbsp;&nbsp;&nbsp;y = {a:.4f} × {x_bar:.1f} + {b:.1f} = <b>{a*x_bar+b:.1f}</b>　← 應等於 ȳ = {y_bar:.1f} ✓<br><br>
驗算 2：用 sklearn 算出的斜率
</div>""", unsafe_allow_html=True)
        from sklearn.linear_model import LinearRegression as LR
        _m = float(LR().fit(sub[[x_col]], sub["統測_總分數"]).coef_[0])
        _b = float(LR().fit(sub[[x_col]], sub["統測_總分數"]).intercept_)
        c1, c2 = st.columns(2)
        c1.metric("手算斜率 a", f"{a:.4f}")
        c2.metric("sklearn 斜率 a", f"{_m:.4f}")
        c1.metric("手算截距 b", f"{b:.1f}")
        c2.metric("sklearn 截距 b", f"{_b:.1f}")
        st.success("✅ 兩者完全一致——公式和程式算的是同一件事！")

        # 抽取 5 筆展示計算細節
        st.markdown("#### 抽 5 筆資料看 (xi - x̄) 和 (yi - ȳ) 的計算")
        sample = sub.sample(5, random_state=7).copy()
        sample["xi − x̄"] = (sample[x_col] - x_bar).round(2)
        sample["yi − ȳ"]  = (sample["統測_總分數"] - y_bar).round(2)
        sample["(xi−x̄)(yi−ȳ)"] = (sample["xi − x̄"] * sample["yi − ȳ"]).round(2)
        sample["(xi−x̄)²"]      = (sample["xi − x̄"] ** 2).round(2)
        sample = sample.rename(columns={x_col: f"模{exam}分", "統測_總分數": "統測分"})
        st.dataframe(sample[[f"模{exam}分", "統測分", "xi − x̄",
                              "yi − ȳ", "(xi−x̄)(yi−ȳ)", "(xi−x̄)²"]],
                     hide_index=True)


# ── 活動 8：IT 老師 ────────────────────────────────────────────────

def s16():  # 提問 + 初始散佈圖
    st.markdown('<span class="tag">活動 8 · 資訊老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">EDA 讓我們看見過去<br>但「預測」才是資訊最迷人的地方</h2>',
                unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="margin-top:32px">
問題：<b>能不能用前幾次模擬考的成績，預測你最終的統測分數？</b><br><br>
先來看模考和統測的分布關係……
</div>""", unsafe_allow_html=True)
    sub = wide[["模5_總分數","統測_總分數"]].dropna()
    fig = px.scatter(sub, x="模5_總分數", y="統測_總分數", opacity=0.3,
                     labels={"模5_總分數":"第5次模考總分","統測_總分數":"統測總分"},
                     color_discrete_sequence=["#4b9eff"])
    fig.update_layout(font_size=18, height=430)
    st.plotly_chart(fig, width="stretch")
    st.caption("散佈圖：每個點是一位學長姐。是否有某種規律？")


def s17():  # 相關係數互動
    st.markdown('<span class="tag">活動 8 · 資訊老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">換不同模考，相關係數 r 怎麼變？</h2>', unsafe_allow_html=True)

    exam = st.slider("選第幾次模擬考", 1, 5, 5, key="s17e")
    x_col = f"模{exam}_總分數"
    sub = wide[[x_col, "統測_總分數"]].dropna()
    r = sub.corr().iloc[0, 1]
    _ols = LinearRegression().fit(sub[[x_col]], sub["統測_總分數"])
    xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 60)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("相關係數 r", f"{r:.3f}")
    c2.metric("樣本數 n", f"{len(sub):,}")
    c3.metric("斜率 a",  f"{float(_ols.coef_[0]):.3f}")
    c4.metric("截距 b",  f"{float(_ols.intercept_):.1f}")

    fig = px.scatter(sub, x=x_col, y="統測_總分數", opacity=0.25,
                     labels={x_col: f"第{exam}次模考總分", "統測_總分數": "統測總分"})
    fig.add_scatter(x=xs, y=_ols.coef_[0]*xs+_ols.intercept_,
                    mode="lines", name="OLS 趨勢線",
                    line=dict(color="#ff4b4b", width=4))
    fig.update_layout(font_size=18, height=460)
    st.plotly_chart(fig, width="stretch")
    st.info(f"💡 從模1到模5，r 值通常會逐漸上升——離統測越近，預測越準確。")


def s18():  # 預測實機
    st.markdown('<span class="tag">活動 8 · 預測實機</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">丟入模考分數，預測統測落點</h2>', unsafe_allow_html=True)

    exam = st.slider("選第幾次模擬考", 1, 5, 5, key="s18e")
    x_col = f"模{exam}_總分數"
    sub = wide[[x_col, "統測_總分數"]].dropna()
    _ols = LinearRegression().fit(sub[[x_col]], sub["統測_總分數"])
    m_ols, b_ols = float(_ols.coef_[0]), float(_ols.intercept_)

    c_input, c_result = st.columns([1, 2])
    with c_input:
        st.markdown("#### 輸入假想學生的模考分數")
        scores = {}
        for label, default in [("同學甲", 250), ("同學乙", 320), ("同學丙", 400)]:
            scores[label] = st.number_input(label, 0, 500, default, step=10, key=f"s18_{label}")

    with c_result:
        xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 60)
        fig = px.scatter(sub, x=x_col, y="統測_總分數", opacity=0.2,
                         labels={x_col:f"第{exam}次模考總分","統測_總分數":"統測總分"})
        fig.add_scatter(x=xs, y=m_ols*xs+b_ols, mode="lines",
                        name="迴歸線", line=dict(color="#ff4b4b", width=3))
        colors = ["#ffcc00","#4bff91","#4b9eff"]
        for (label, sc), color in zip(scores.items(), colors):
            pred = m_ols * sc + b_ols
            fig.add_scatter(x=[sc], y=[pred], mode="markers+text",
                            name=label,
                            text=[f"{label}≈{pred:.0f}分"],
                            textposition="top center",
                            marker=dict(size=18, color=color, symbol="star"))
        fig.update_layout(font_size=17, height=460)
        st.plotly_chart(fig, width="stretch")

    st.markdown(f"**迴歸方程式：統測總分 ≈ {m_ols:.2f} × 模{exam}總分 + {b_ols:.1f}**")


# ── 活動 9：數學老師 ───────────────────────────────────────────────

def s19():  # y=ax+b 解說
    st.markdown('<span class="tag">活動 9 · 數學老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">拆解黑盒子：那條線從哪裡來？</h2>', unsafe_allow_html=True)
    c_eq, c_chart = st.columns([1, 2])
    with c_eq:
        st.markdown("""
<div class="big">
程式跑出的那條線就是：<br><br>
<div style="font-size:42px;font-weight:bold;color:#ff4b4b;text-align:center;padding:20px 0">
y = ax + b
</div>
<b>x</b> = 模考總分（輸入）<br>
<b>y</b> = 統測總分（預測）<br>
<b>a</b> = 斜率（每多 1 分模考<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;統測預期多幾分）<br>
<b>b</b> = 截距（x=0 時的 y）<br><br>
這就是你國中學過的<br><b>一次函數</b>！
</div>""", unsafe_allow_html=True)
    with c_chart:
        sub = wide[["模5_總分數","統測_總分數"]].dropna()
        _ols = LinearRegression().fit(sub[["模5_總分數"]], sub["統測_總分數"])
        m, b = float(_ols.coef_[0]), float(_ols.intercept_)
        xs = np.linspace(sub["模5_總分數"].min(), sub["模5_總分數"].max(), 60)
        fig = px.scatter(sub, x="模5_總分數", y="統測_總分數", opacity=0.2,
                         labels={"模5_總分數":"模考總分","統測_總分數":"統測總分"})
        fig.add_scatter(x=xs, y=m*xs+b, mode="lines",
                        name=f"y = {m:.2f}x + {b:.1f}",
                        line=dict(color="#ff4b4b", width=4))
        fig.update_layout(font_size=17, height=480, showlegend=True)
        st.plotly_chart(fig, width="stretch")


def s20():  # 你來畫迴歸線
    st.markdown('<span class="tag">活動 9 · 互動挑戰</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">你來畫一條預測線試試看</h2>', unsafe_allow_html=True)
    st.caption("拉兩個點決定一條直線，看看你的 SSE 能不能比 OLS 更小？")

    exam = st.select_slider("選模考次數", options=list(range(1,6)), value=5, key="s20e")
    x_col = f"模{exam}_總分數"
    sub = wide[[x_col,"統測_總分數"]].dropna()
    x_min, x_max = int(sub[x_col].min()), int(sub[x_col].max())
    y_min, y_max = int(sub["統測_總分數"].min()), int(sub["統測_總分數"].max())

    _ols = LinearRegression().fit(sub[[x_col]], sub["統測_總分數"])
    m_ols, b_ols = float(_ols.coef_[0]), float(_ols.intercept_)
    sse_ols = float(((sub["統測_總分數"] - _ols.predict(sub[[x_col]]))**2).sum())

    c1, c2 = st.columns(2)
    p1x = c1.slider("第 1 點 x（模考）", x_min, x_max, x_min+40, key="s20p1x")
    p1y = c1.slider("第 1 點 y（統測）", y_min, y_max, y_min+40, key="s20p1y")
    p2x = c2.slider("第 2 點 x", x_min, x_max, x_max-40, key="s20p2x")
    p2y = c2.slider("第 2 點 y", x_min, y_max, y_max-40, key="s20p2y")

    m_user = (p2y - p1y) / max(p2x - p1x, 1e-6)
    b_user = p1y - m_user * p1x
    sse_user = float(((sub["統測_總分數"] - (m_user*sub[x_col]+b_user))**2).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("你的線 SSE", f"{sse_user:,.0f}")
    m2.metric("OLS 最佳解 SSE", f"{sse_ols:,.0f}")
    delta = sse_user - sse_ols
    m3.metric("你比 OLS 多了", f"{delta:,.0f}",
              delta_color="inverse" if delta > 0 else "normal")

    xs = np.linspace(x_min, x_max, 60)
    fig = px.scatter(sub, x=x_col, y="統測_總分數", opacity=0.2,
                     labels={x_col:f"模{exam}總分","統測_總分數":"統測總分"})
    fig.add_scatter(x=xs, y=m_ols*xs+b_ols, mode="lines",
                    name="OLS 最佳解", line=dict(color="#ff4b4b",width=3))
    fig.add_scatter(x=xs, y=m_user*xs+b_user, mode="lines",
                    name="你的線", line=dict(color="#ffcc00",width=3,dash="dash"))
    fig.add_scatter(x=[p1x,p2x], y=[p1y,p2y], mode="markers",
                    name="你拉的兩點", marker=dict(size=16,color="#ffcc00",symbol="x"))
    fig.update_layout(font_size=17, height=420)
    st.plotly_chart(fig, width="stretch")
    st.info("💡 OLS 的 SSE 是所有可能直線中最小的——不管怎麼拉，你的 SSE 永遠 ≥ OLS。"
            "這就是「最小平方法」名字的由來。")


def s21():  # 殘差視覺化
    st.markdown('<span class="tag">活動 9 · 數學老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">殘差（誤差）長什麼樣子？</h2>', unsafe_allow_html=True)
    st.caption("每條紅線 = 真實值 − 預測值 = 殘差；SSE = 所有殘差² 的總和")

    sub_full = wide[["模5_總分數","統測_總分數"]].dropna().reset_index(drop=True)
    _ols = LinearRegression().fit(sub_full[["模5_總分數"]], sub_full["統測_總分數"])
    sub_full["預測"] = _ols.predict(sub_full[["模5_總分數"]])
    sub_full["殘差"] = sub_full["統測_總分數"] - sub_full["預測"]

    n_show = st.slider("顯示幾個殘差線段（隨機抽樣）", 20, 200, 80, step=10, key="s21n")
    sample = sub_full.sample(n_show, random_state=42)

    xs = np.linspace(sub_full["模5_總分數"].min(), sub_full["模5_總分數"].max(), 60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub_full["模5_總分數"], y=sub_full["統測_總分數"],
        mode="markers", opacity=0.15, name="實際資料",
        marker=dict(color="#4b9eff", size=5)))
    fig.add_trace(go.Scatter(
        x=xs, y=_ols.coef_[0]*xs+_ols.intercept_,
        mode="lines", name="OLS 迴歸線",
        line=dict(color="#ff4b4b", width=3)))
    # 殘差線段
    for _, row in sample.iterrows():
        fig.add_shape(type="line",
                      x0=row["模5_總分數"], y0=row["預測"],
                      x1=row["模5_總分數"], y1=row["統測_總分數"],
                      line=dict(color="rgba(255,200,0,0.6)", width=1.5))
    fig.update_layout(font_size=17, height=500,
                      xaxis_title="模5總分", yaxis_title="統測總分",
                      showlegend=True)
    st.plotly_chart(fig, width="stretch")

    sse = float((sub_full["殘差"]**2).sum())
    rmse = float((sub_full["殘差"]**2).mean()**0.5)
    c1, c2, c3 = st.columns(3)
    c1.metric("SSE（殘差平方和）", f"{sse:,.0f}")
    c2.metric("RMSE（均方根誤差）", f"{rmse:.1f} 分")
    c3.metric("平均誤差方向", f"{'正偏' if sub_full['殘差'].mean()>0 else '負偏'} {abs(sub_full['殘差'].mean()):.1f} 分")


_W_PAT  = re.compile(r"(國文|英文|數學|專業\(一\)|專業\(二\))\*(\d+\.?\d*)")
_W_KEY  = {"國文":"國文","英文":"英文","數學":"數學","專業(一)":"專一","專業(二)":"專二"}

def _parse_w(formula):
    return {_W_KEY[k]: float(v) for k, v in _W_PAT.findall(str(formula))}

def _calc_w(scores, weights):
    return sum(scores.get(s, 0) * w for s, w in weights.items())


def s21b():  # 科目分數 → 加權落點
    from data_loader import load_thresholds
    st.markdown('<span class="tag">活動 9 · 雙師協同</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">輸入五科分數，找出你的升學落點</h2>', unsafe_allow_html=True)
    st.caption("每個學校都有自己的加權公式——同樣的原始分數，換一間學校結果就不同！")

    c_score, c_filter, c_out = st.columns([1, 1, 2])

    SUBJ_DEFAULTS = {"國文": 70, "英文": 65, "數學": 60, "專一": 72, "專二": 68}
    scores = {}
    with c_score:
        st.markdown("##### 輸入各科預估分數（0–100）")
        dept_sel = st.selectbox("套用科別中位數", ["（自行輸入）"] + DEPT_ORDER,
                                key="s21b_dept")
        if dept_sel != "（自行輸入）":
            for sname, col in SUBJ.items():
                med = wide[wide["科別"] == dept_sel][col].median()
                if pd.notna(med):
                    SUBJ_DEFAULTS[sname] = int(med)

        for sname in ["國文","英文","數學","專一","專二"]:
            scores[sname] = st.number_input(sname, 0, 100,
                                             value=SUBJ_DEFAULTS[sname],
                                             step=1, key=f"s21b_{sname}")
        raw_total = sum(scores.values())
        st.markdown(f"""
<div style="padding:12px;background:#222;border-radius:8px;text-align:center;margin-top:8px">
  <div style="color:#888;font-size:13px">原始總分</div>
  <div style="font-size:44px;font-weight:bold;color:#fff">{raw_total}</div>
  <div style="color:#555;font-size:12px">滿分 500</div>
</div>""", unsafe_allow_html=True)

    with c_filter:
        st.markdown("##### 查詢條件")
        cat    = st.radio("科系分類", ["商管類","資訊類","語文類","設計類"],
                          key="s21b_cat")
        year   = st.selectbox("查哪年門檻", [114, 113, 112], key="s21b_yr")
        margin = st.slider("門檻 ± 分差", 10, 80, 30, step=5, key="s21b_mg")

        st.markdown("---")
        st.markdown("""
<div style="font-size:14px;color:#aaa">
💡 <b>加權分怎麼算？</b><br>
各科分數 × 學校指定倍率再加總<br>
例：國文×1 + 英文×1.5 + 數學×1.5<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 專一×2 + 專二×3
</div>""", unsafe_allow_html=True)

    with c_out:
        thresh = load_thresholds()
        th_sub = thresh[
            (thresh["年度"] == year) &
            (thresh["科系分類"] == cat)
        ].copy()

        def row_w(row):
            try:
                return round(_calc_w(scores, _parse_w(row["各科目加權"])), 2)
            except Exception:
                return None

        th_sub["你的加權分"] = th_sub.apply(row_w, axis=1)
        th_sub["分差"] = (th_sub["錄取總分數"] - th_sub["你的加權分"]).round(1)
        th_sub["最高可能分"] = th_sub["各科目加權"].apply(
            lambda f: round(sum(_parse_w(f).values()) * 100)
            if pd.notna(f) else None
        )

        hit = th_sub[th_sub["分差"].abs() <= margin].dropna(subset=["你的加權分"])
        hit_sorted = hit.sort_values("分差", key=abs)

        if hit_sorted.empty:
            st.warning(f"± {margin} 分差內無符合的 {year} 年 {cat} 志願。")
        else:
            st.caption(f"{year} 年 {cat}｜± {margin} 分｜共 **{len(hit_sorted)}** 筆")
            show_cols = [c for c in ["學校名稱","系科組學程名稱","學校類型",
                                      "你的加權分","錄取總分數","分差","最高可能分"]
                         if c in hit_sorted.columns]
            st.dataframe(hit_sorted[show_cols].head(20),
                         hide_index=True, height=430)
        st.caption("分差 > 0 表示門檻比你高（差多少分才夠）；< 0 表示你超過門檻。")


def s21c():  # 推甄 vs 分發
    from data_loader import load_thresholds
    st.markdown('<span class="tag">活動 9 · 數學老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">推甄 vs 分發：誰能上更好的學校？</h2>',
                unsafe_allow_html=True)
    st.caption("分發：你的加權分 ≥ 門檻才能錄取　｜　推甄：分數低於門檻也有機會——靠學業成績、備審、面試")

    c_score, c_out = st.columns([1, 3])

    with c_score:
        st.markdown("##### 輸入各科預估分數")
        dept_sel = st.selectbox("套用科別中位數", ["（自行輸入）"] + DEPT_ORDER, key="s21c_dept")
        DEFS = {"國文": 70, "英文": 65, "數學": 60, "專一": 72, "專二": 68}
        if dept_sel != "（自行輸入）":
            for sname, col in SUBJ.items():
                med = wide[wide["科別"] == dept_sel][col].median()
                if pd.notna(med):
                    DEFS[sname] = int(med)
        scores = {}
        for sname in ["國文", "英文", "數學", "專一", "專二"]:
            scores[sname] = st.number_input(sname, 0, 100,
                                             value=DEFS[sname], step=1,
                                             key=f"s21c_{sname}")
        cat  = st.radio("科系分類", ["商管類","資訊類","語文類","設計類"], key="s21c_cat")
        year = st.selectbox("查哪年門檻", [114, 113, 112], key="s21c_yr")
        margin_rec = st.slider("推甄機會範圍（門檻超過你幾分以內）",
                               10, 100, 50, step=10, key="s21c_mg")

    with c_out:
        thresh = load_thresholds()
        th_sub = thresh[(thresh["年度"] == year) & (thresh["科系分類"] == cat)].copy()

        def row_w(row):
            try:
                return round(_calc_w(scores, _parse_w(row["各科目加權"])), 2)
            except Exception:
                return None

        th_sub["你的加權分"] = th_sub.apply(row_w, axis=1)
        th_sub = th_sub.dropna(subset=["你的加權分"])
        th_sub["分差"] = (th_sub["錄取總分數"] - th_sub["你的加權分"]).round(1)

        # 分類
        def classify(diff):
            if diff <= 0:
                return "分發可上 ✅"
            elif diff <= margin_rec:
                return "推甄機會 🎯"
            else:
                return "差距較大"
        th_sub["情境"] = th_sub["分差"].apply(classify)

        n_dist  = (th_sub["情境"] == "分發可上 ✅").sum()
        n_rec   = (th_sub["情境"] == "推甄機會 🎯").sum()
        dist_best = th_sub.loc[th_sub["情境"]=="分發可上 ✅","錄取總分數"].max() if n_dist else 0
        rec_best  = th_sub.loc[th_sub["情境"]=="推甄機會 🎯","錄取總分數"].max() if n_rec  else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("分發可上 志願數", n_dist)
        m2.metric("推甄機會 志願數", n_rec)
        m3.metric("分發最高門檻", f"{dist_best:.0f}" if dist_best else "—")
        m4.metric("推甄可挑戰最高", f"{rec_best:.0f}" if rec_best else "—",
                  delta=f"+{rec_best-dist_best:.0f}" if rec_best > dist_best else None)

        # 視覺化：水平點圖，學校依門檻排序
        plot_df = th_sub[th_sub["情境"] != "差距較大"].copy()
        plot_df = plot_df.nlargest(50, "錄取總分數")
        plot_df["學校科系"] = (plot_df["學校名稱"].str[:8] + " " +
                              plot_df["系科組學程名稱"].str[:6])
        color_map = {"分發可上 ✅": "#4bff91", "推甄機會 🎯": "#ffcc00"}
        fig = px.scatter(
            plot_df.sort_values("錄取總分數"),
            x="錄取總分數", y="學校科系",
            color="情境", color_discrete_map=color_map,
            hover_data=["你的加權分","分差"],
            labels={"錄取總分數": "錄取加權門檻", "學校科系": ""},
        )
        fig.update_traces(marker_size=14)
        fig.update_layout(
            font_size=13,
            height=max(380, len(plot_df) * 22 + 80),
            legend=dict(orientation="h", y=1.03),
            margin=dict(l=0, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, width="stretch")

        if rec_best > dist_best:
            st.success(
                f"💡 推甄可以挑戰比分發上限高 **{rec_best-dist_best:.0f} 分** 的學校！"
                f"（分發上限 {dist_best:.0f}，推甄可挑戰 {rec_best:.0f}）")
        elif n_dist > 0:
            st.info("你的分數已能分發進不錯的學校，推甄還可挑戰競爭更激烈的科系。")

        tab_dist, tab_rec = st.tabs(
            ["✅ 分發可上（加權分 ≥ 門檻）",
             f"🎯 推甄機會（門檻比你高 1–{margin_rec} 分）"])
        with tab_dist:
            df_d = th_sub[th_sub["情境"]=="分發可上 ✅"].sort_values(
                "錄取總分數", ascending=False)
            st.dataframe(
                df_d[["學校名稱","系科組學程名稱","學校類型","你的加權分","錄取總分數","分差"]].head(15),
                hide_index=True, height=300)
        with tab_rec:
            df_r = th_sub[th_sub["情境"]=="推甄機會 🎯"].sort_values(
                "錄取總分數", ascending=False)
            st.dataframe(
                df_r[["學校名稱","系科組學程名稱","學校類型","你的加權分","錄取總分數","分差"]].head(15),
                hide_index=True, height=300)


def s22():  # 總結：打破預測
    st.markdown("""
<div class="full center">
<span class="tag">活動 9 · 雙師總結</span><br><br>
<h1 class="st">預測分數低，不用灰心</h1>
<div class="big" style="margin-top:40px">
模型只是根據<b>過去的規律</b>預測——<br>
但你是一個有能力<b>改變自己斜率</b>的人。<br><br>
努力，讓你成為散佈圖上那個<br>
<span style="color:#ff4b4b;font-size:32px">向上突破的離群值 ⬆</span>
</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
SLIDE_FUNCS = [
    s0, s1, s2, s3, s4, s5, s6,                         # 0–6
    s7, s8a, s8b, s9,                                    # 7–10
    s10, s11, s12, s13, s14,                             # 11–15
    s15, s15b, s16, s17, s18, s19, s20, s21, s21b, s21c, s22,  # 16–26
]
N = len(SLIDE_FUNCS)

SLIDE_FUNCS[slide]()
nav_bar(slide, N)
