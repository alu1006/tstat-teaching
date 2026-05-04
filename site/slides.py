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
import gsheet

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
/* sidebar */
[data-testid="collapsedControl"] { visibility: visible !important; }
section[data-testid="stSidebar"] > div:first-child { min-width: 220px; }
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
    ("①", "薪資真相（小組）",    ""),
    ("①", "統測例題：中位數",    ""),
    ("②", "第二階段說明",        "phase"),
    ("②", "折線圖",              ""),
    ("②", "統測來了",            ""),
    ("②", "長條圖",              ""),
    ("②", "Q1/Q2/Q3/IQR",       ""),
    ("②", "箱型圖",              ""),
    ("②", "EDA 總結",            ""),
    ("③", "第三階段說明",        "phase"),
    ("③", "正相關 vs 負相關",      ""),
    ("③", "相關係數 r 怎麼算",    ""),
    ("③", "換你算：5 筆小資料",  ""),
    ("③", "分組討論計時",        ""),
    ("④", "第四階段說明",        "phase"),
    ("④", "提問：能預測嗎？",    ""),
    ("④", "迴歸直線怎麼算？",   ""),
    ("④", "最小平方法 (OLS)",    ""),
    ("④", "模考→統測→落點",      ""),
    ("④", "落點之後·個人反思",   ""),
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
        "📣","🔄","🧮","📈","🎉","📊",
        "📐","📦","📣","📐","🧐",
        "📣","📐","🤔","✏️","🔮","📈","🎯","🎯","⚖️","🏆",
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
    view = st.radio("檢視模式", ["📋 目錄", "🖼 縮圖"],
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
  <span style="flex:1"></span>
  <span class="snav-pg" style="flex:2;text-align:center">{s+1} / {n}　{icon} {title}</span>
  <span style="flex:1"></span>
</div>
<style>
div[data-testid="stHorizontalBlock"]:last-of-type button {{
    height: 48px; font-size: 18px; font-weight: bold;
}}
/* 導覽列中的 selectbox：深色背景、白色文字 */
div[data-testid="stHorizontalBlock"]:last-of-type div[data-baseweb="select"] > div {{
    background: #1e2130 !important;
    border-color: #444 !important;
    color: #ddd !important;
    font-size: 14px !important;
}}
</style>
""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 5, 2])
    with col1:
        if s > 0:
            if st.button("← 上一頁", key="nav_prev", use_container_width=True):
                st.session_state.slide = s - 1; st.rerun()
    with col2:
        jump_options = [f"{i+1}. {icon} {t}" for i, (icon, t, _) in enumerate(SLIDES_META)]
        def _on_jump():
            sel = st.session_state["nav_jump"]
            st.session_state.slide = jump_options.index(sel)
        st.session_state["nav_jump"] = jump_options[s]
        st.selectbox("跳至", jump_options, key="nav_jump",
                     label_visibility="collapsed", on_change=_on_jump)
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


def s1():  # 第一階段：開場引言
    st.markdown('<h1 class="st">📣 開場：來看看學長姐的真實戰果</h1>', unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="line-height:2.0">
<br>
🎓 &nbsp;<b>115 學年度統測剛結束</b>——再過幾個月就是你們了。<br><br>

但是在你們進考場之前，先看一份「劇透」：<br>
<span style="color:#ff4b4b;font-weight:bold">112、113、114 三屆學長姐</span>
留下了完整的模考 + 統測成績資料。<br><br>

📌 &nbsp;我們要一起回答兩個問題：<br>
　　1. <b>哪一科才是真正的「大魔王」？</b>（你以為的最低，跟實際一樣嗎？）<br>
　　2. <b>「平均分數」這個數字，到底有多容易騙人？</b>
</div>""", unsafe_allow_html=True)


def s2():  # 原始資料
    st.markdown('<span class="tag">活動 1 · 資訊老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">這就是「原始資料」</h2>', unsafe_allow_html=True)
    st.caption(f"共 {len(wide):,} 位學長姐 · 記錄從第一次模擬考到統測的完整成績 · 每一列就是一個人")

    c_link1, c_link2, _ = st.columns([1, 1, 4])
    c_link1.link_button(
        "🚀 開啟 Colab 練習",
        "https://colab.research.google.com/drive/1dn7WgORYXipo0e5t8oYp1TSWA9vSYDe4#scrollTo=rFFw75NmiObd",
        use_container_width=True,
    )
    c_link2.link_button(
        "📁 原始資料雲端",
        "https://drive.google.com/file/d/1_eD1Ozv0Z6F6X0xbGLOyollLb_VQvzD_/view?usp=sharing",  # 01_wide_scores.csv
        use_container_width=True,
    )

    st.dataframe(wide.drop(columns=["科別"]), height=480, use_container_width=True)
    st.markdown("""<div class="big" style="margin-top:16px">
光用眼睛看，你能找出「哪一科的平均分數最低」嗎？
</div>""", unsafe_allow_html=True)


def _persist_voter_state():
    """避免 Streamlit 在切頁時把 widget state 清掉——把 voter_xxx 重新指派給自己。
    這個 trick 必須在每次 script run 的最頂端跑（不只是有 widget 的頁）。
    """
    for k in ("voter_group", "voter_sid", "voter_name"):
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def _voter_inputs(prefix: str = ""):
    """渲染 3 個身分輸入欄；所有頁共用同一個 widget key，狀態自動跨頁同步。"""
    c_g, c_id, c_name = st.columns([1, 2, 2])
    g = c_g.text_input("第幾組", key="voter_group", placeholder="例：3")
    s = c_id.text_input("學號", key="voter_sid", placeholder="例：1130123")
    n = c_name.text_input("姓名", key="voter_name", placeholder="例：王小明")
    return g, s, n


def s3():  # 投票（Google Sheet 連線版）
    _persist_voter_state()
    st.markdown('<span class="tag">活動 1 · 即時投票</span>', unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center;padding-top:4vh">
<h1 class="st">你覺得統測哪一科<br>全班平均最低？</h1>
</div>""", unsafe_allow_html=True)

    QKEY = "lowest_subject"
    options = list(SUBJ.keys())
    counts = gsheet.get_votes(QKEY, options)
    online = gsheet.is_connected()

    group, sid, name = _voter_inputs("s3")

    voted = bool(sid) and gsheet.has_voted(QKEY, sid)
    can_vote = bool(group.strip()) and bool(sid.strip()) and bool(name.strip()) and not voted

    if voted:
        st.warning(f"⚠️ 學號 {sid} 已經投過票了，每人只能投一次")
    elif not can_vote:
        st.info("👆 請先填好 第幾組／學號／姓名 才能投票")

    cols = st.columns(5)
    for s, col in zip(options, cols):
        cnt = counts.get(s, 0)
        if col.button(
            f"**{s}**\n\n{cnt} 票",
            key=f"v3_{s}",
            use_container_width=True,
            disabled=not can_vote,
        ):
            ok, msg = gsheet.add_vote(QKEY, s, group=group, student_id=sid, name=name)
            if ok:
                st.toast(f"✅ {name} 投給 {s}")
            else:
                st.error(msg)
            st.rerun()

    status_text = (
        "🟢 已連結 Google Sheet · 全班即時同步"
        if online
        else "🟡 未連結 Google Sheet（fallback：本機 session）"
    )
    st.markdown(
        f"<div style='text-align:center;color:#666;font-size:14px;margin-top:8px'>{status_text}</div>",
        unsafe_allow_html=True,
    )

    c_reveal, c_reset = st.columns([3, 1])
    if c_reveal.button("🔍 揭曉答案", key="reveal3", use_container_width=True):
        st.session_state.revealed3 = True
    if c_reset.button("♻ 清除投票", key="reset3", use_container_width=True):
        gsheet.reset_votes(QKEY)
        st.session_state.revealed3 = False
        st.rerun()

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
        st.code(
            'subj_cols = [\n'
            '    "統測_國文分數", "統測_英文分數",\n'
            '    "統測_數學B分數",\n'
            '    "統測_專一分數", "統測_專二分數",\n'
            ']\n'
            'df[subj_cols].mean().sort_values()',
            language="python",
        )
        st.dataframe(pd.DataFrame(avgs.items(), columns=["科目","平均分數"])
                       .sort_values("平均分數"), hide_index=True, height=260)
    with c_chart:
        df = pd.DataFrame(avgs.items(), columns=["科目","平均分數"]).sort_values("平均分數")
        fig = px.bar(df, x="科目", y="平均分數", color="科目", text="平均分數",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_traces(textposition="outside", textfont_size=22)
        fig.update_layout(font_size=18, height=500, showlegend=False, yaxis_range=[0, 115])
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("✨ 在 Colab 中直接問 AI ✨", expanded=False):
        st.markdown(
            "**情境**：我有一個 pandas DataFrame `df`,裡面有統測五科分數欄位。\n\n"
            "**請幫我**:算出每一科的平均分數,由低到高排序,並畫成長條圖。"
        )
        st.code(
            '我有一個 pandas DataFrame 叫 df,欄位包含:\n'
            '"統測_國文分數"、"統測_英文分數"、"統測_數學B分數"、\n'
            '"統測_專一分數"、"統測_專二分數"。\n\n'
            '請幫我用 Python 做兩件事:\n'
            '1. 算出這五個科目的平均分數,由低到高排序,印出來。\n'
            '2. 用 plotly 把結果畫成長條圖,x 軸是科目、y 軸是平均分數,\n'
            '   每根長條上方顯示數值。',
            language="text",
        )
        st.caption("💡 小技巧:在 Colab 儲存格按 ✨ 圖示就能直接問 Gemini,把欄位名稱、圖表種類、輸出格式講清楚,AI 一次就給對答案。")


def s5():  # 陷阱題
    _persist_voter_state()
    avgs = {s: wide[col].mean() for s, col in SUBJ.items()}
    lowest_s, lowest_v = min(avgs, key=avgs.get), round(min(avgs.values()), 1)
    st.markdown('<span class="tag">活動 2 · 數學老師</span>', unsafe_allow_html=True)
    st.markdown(f"""
<div style="text-align:center;padding-top:2vh">
<h1 class="st">{lowest_s} 的平均是 {lowest_v} 分<br><br>
那「中位數」會比平均數<br>
<span style="color:#ff4b4b">高</span> 還是 <span style="color:#4b9eff">低</span>？
</h1></div>""", unsafe_allow_html=True)

    QKEY = "median_vs_mean"
    options = ["高", "低"]
    counts = gsheet.get_votes(QKEY, options)
    online = gsheet.is_connected()

    group, sid, name = _voter_inputs("s5")

    voted = bool(sid) and gsheet.has_voted(QKEY, sid)
    can_vote = bool(group.strip()) and bool(sid.strip()) and bool(name.strip()) and not voted

    if voted:
        st.warning(f"⚠️ 學號 {sid} 已經投過票了，每人只能投一次")
    elif not can_vote:
        st.info("👆 請先填好 第幾組／學號／姓名 才能投票")

    c1, c2 = st.columns(2)
    hi = counts.get("高", 0)
    lo = counts.get("低", 0)
    if c1.button(
        f"🙋  比較**高**　　{hi} 人",
        key="v5h",
        use_container_width=True,
        disabled=not can_vote,
    ):
        ok, msg = gsheet.add_vote(QKEY, "高", group=group, student_id=sid, name=name)
        if ok:
            st.toast(f"✅ {name} 投了「高」")
        else:
            st.error(msg)
        st.rerun()
    if c2.button(
        f"🙋  比較**低**　　{lo} 人",
        key="v5l",
        use_container_width=True,
        disabled=not can_vote,
    ):
        ok, msg = gsheet.add_vote(QKEY, "低", group=group, student_id=sid, name=name)
        if ok:
            st.toast(f"✅ {name} 投了「低」")
        else:
            st.error(msg)
        st.rerun()

    status_text = (
        "🟢 已連結 Google Sheet · 全班即時同步"
        if online
        else "🟡 未連結 Google Sheet（fallback：本機 session）"
    )
    st.markdown(
        f"<div style='text-align:center;color:#666;font-size:14px;margin-top:8px'>{status_text}</div>",
        unsafe_allow_html=True,
    )


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
        # 平均數：紅色虛線，標註固定在「上方」
        fig.add_vline(
            x=mean_v, line_dash="dash", line_color="#ff4b4b", line_width=3,
            annotation_text=f"<b>📊 平均數 = {mean_v:.1f}</b>",
            annotation_font=dict(size=18, color="#ffffff"),
            annotation_bgcolor="#ff4b4b",
            annotation_bordercolor="#ff4b4b",
            annotation_borderwidth=2,
            annotation_borderpad=6,
            annotation_position="top left",
        )
        # 中位數：綠色實線，標註固定在「下方」
        fig.add_vline(
            x=med_v, line_color="#22c55e", line_width=3,
            annotation_text=f"<b>📍 中位數 = {med_v:.1f}</b>",
            annotation_font=dict(size=18, color="#ffffff"),
            annotation_bgcolor="#22c55e",
            annotation_bordercolor="#22c55e",
            annotation_borderwidth=2,
            annotation_borderpad=6,
            annotation_position="bottom right",
        )
        fig.update_layout(font_size=18, height=520, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    diff = med_v - mean_v
    direction = "左偏（負偏）" if diff > 0 else "右偏（正偏）"
    with c_exp:
        st.markdown(f"""
<div class="big">
<div style="background:#ff4b4b;color:#fff;padding:10px;border-radius:8px;margin-bottom:10px">
  <span style="font-size:14px">📊 紅色虛線</span><br>
  <b>平均數</b><br>{mean_v:.1f}
</div>
<div style="background:#22c55e;color:#fff;padding:10px;border-radius:8px;margin-bottom:10px">
  <span style="font-size:14px">📍 綠色實線</span><br>
  <b>中位數</b><br>{med_v:.1f}
</div>
<b>差距</b>　{abs(diff):.1f} 分<br>
<span style="color:#ff4b4b">{direction}</span><br><br>
<span style="font-size:14px;color:#aaa">少數極低分把平均往下拖</span>
</div>""", unsafe_allow_html=True)


def s_pay():  # 薪資中位數 vs 平均數小組討論
    st.markdown('<span class="tag">活動 2 · 小組討論</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">真實案例：台灣薪資的真相</h2>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div style="background:#fff3f3;padding:24px;border-radius:12px;text-align:center;border:2px solid #ff4b4b">
  <div style="font-size:18px;color:#666">📊 平均薪資</div>
  <div style="font-size:52px;font-weight:bold;color:#ff4b4b;margin:8px 0">47,884 元</div>
  <div style="font-size:14px;color:#888">每月經常性薪資</div>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div style="background:#f3f7ff;padding:24px;border-radius:12px;text-align:center;border:2px solid #4b9eff">
  <div style="font-size:18px;color:#666">📐 中位數薪資</div>
  <div style="font-size:52px;font-weight:bold;color:#4b9eff;margin:8px 0">38,406 元</div>
  <div style="font-size:14px;color:#888">較能反映中低薪族群感受</div>
</div>
""", unsafe_allow_html=True)

    gap = 47884 - 38406
    st.markdown(f"""
<div style="text-align:center;margin-top:24px;font-size:22px">
平均比中位數高 <b style="color:#ff4b4b">{gap:,} 元</b>
</div>""", unsafe_allow_html=True)

    st.markdown("""
---
### 🗣️ 小組討論（2 分鐘）

1. 剛才直方圖：平均**比中位數低**，是因為少數人考很差把平均**拖下去**
2. 這次薪資反過來：**平均比中位數高**——
   - 這代表台灣多數上班族的薪水，是**比平均高**還是**比平均低**？
   - 為什麼會這樣？哪一群人會把平均「拉上去」？
""")

    QKEY = "salary_discussion"
    online = gsheet.is_connected()

    group, sid, name = _voter_inputs("s_pay")

    discussion = st.text_area(
        "✍️ 請輸入你們小組的討論結果",
        key="s_pay_text",
        height=140,
        placeholder="例：客觀：薪資高的人將平均_____；主觀：多數上班族其實比平均低，新聞報的平均薪資不代表多數人的真實感受",
    )

    submitted = bool(sid) and gsheet.has_voted(QKEY, sid)
    can_submit = (
        bool(group.strip())
        and bool(sid.strip())
        and bool(name.strip())
        and bool(discussion.strip())
        and not submitted
    )

    if submitted:
        st.warning(f"⚠️ 學號 {sid} 已經提交過討論了，每人只能交一次")
    elif not can_submit:
        st.info("👆 請填好 第幾組／學號／姓名 + 討論內容才能提交")

    if st.button(
        "📤 提交小組討論",
        key="submit_pay",
        use_container_width=True,
        disabled=not can_submit,
        type="primary",
    ):
        ok, msg = gsheet.add_vote(
            QKEY, discussion.strip(), group=group, student_id=sid, name=name
        )
        if ok:
            st.toast(f"✅ {name} 已提交")
        else:
            st.error(msg)
        st.rerun()

    status_text = (
        "🟢 已連結 Google Sheet · 老師端可即時看到所有小組的討論"
        if online
        else "🟡 未連結 Google Sheet（fallback：本機 session）"
    )
    st.markdown(
        f"<div style='text-align:center;color:#666;font-size:14px;margin-top:8px'>{status_text}</div>",
        unsafe_allow_html=True,
    )


def s6b():  # 統測例題：出生人數中位數
    _persist_voter_state()
    st.markdown('<span class="tag">活動 2 · 例題演練</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">統測例題：出生人數的中位數</h2>', unsafe_allow_html=True)
    st.caption("從國家發展委員會人口推估查詢系統可查詢到出生人數折線圖如下，根據圖表，判斷下列選項何者**錯誤**？")

    # 1970–2007（依台灣戶政公開統計近似）
    birth_old = {
        1970:394015,1971:380439,1972:365749,1973:366942,1974:367823,1975:367647,
        1976:423356,1977:395796,1978:410783,1979:422518,1980:413881,1981:414069,
        1982:405263,1983:383439,1984:371008,1985:346208,1986:309230,1987:314024,
        1988:342031,1989:315299,1990:335618,1991:321932,1992:321632,1993:325613,
        1994:322938,1995:329581,1996:325545,1997:326002,1998:271450,1999:283661,
        2000:305312,2001:260354,2002:247530,2003:227070,2004:216419,2005:205854,
        2006:204459,2007:204414,
    }
    # 2008–2025（題目給定）
    birth_new = {
        2008:198733,2009:191310,2010:166886,2011:196627,2012:229481,2013:199113,
        2014:210383,2015:213598,2016:208440,2017:193844,2018:181601,2019:177767,
        2020:165249,2021:153820,2022:138986,2023:135571,2024:134856,2025:107812,
    }
    all_data = {**birth_old, **birth_new}
    df = pd.DataFrame({"年度": list(all_data.keys()), "出生人數": list(all_data.values())})

    c_chart, c_table = st.columns([2, 1])
    with c_chart:
        fig = px.line(df, x="年度", y="出生人數", markers=True,
                      labels={"年度":"年", "出生人數":"出生人數"})
        fig.update_traces(line_width=2, marker_size=5, marker_color="#4b9eff")
        # 標出三個龍年
        for dragon in [1976, 1988, 2000, 2012]:
            if dragon in all_data:
                fig.add_annotation(x=dragon, y=all_data[dragon],
                                   text=f"🐉{dragon}", showarrow=True, arrowhead=2,
                                   font_size=14, font_color="#ff4b4b", ay=-30)
        fig.update_layout(font_size=15, height=420,
                          xaxis=dict(dtick=3))
        st.plotly_chart(fig, use_container_width=True)

    with c_table:
        st.markdown("##### 表（一）2008–2025")
        tbl = pd.DataFrame({"年度": list(birth_new.keys()),
                            "出生人數": list(birth_new.values())})
        st.dataframe(tbl, hide_index=True, height=420)

    st.markdown("""
<div class="big" style="margin-top:8px">
<b>(A)</b> 1976、1988、2000 這三年（龍年）的出生人數相較於其前一年（兔年）、後一年（蛇年）來得多<br>
<b>(B)</b> 1970 年到 2025 年，出生人數的全距大於 30 萬人<br>
<b>(C)</b> 2008 年到 2016 年，出生人數的中位數為 229481<br>
<b>(D)</b> 2017 年到 2025 年，每一年的出生人數逐年減少
</div>""", unsafe_allow_html=True)

    QKEY = "exam_birth_median"
    options = ["A", "B", "C", "D"]
    counts = gsheet.get_votes(QKEY, options)
    online = gsheet.is_connected()

    group, sid, name = _voter_inputs("s6b")
    voted = bool(sid) and gsheet.has_voted(QKEY, sid)
    can_vote = bool(group.strip()) and bool(sid.strip()) and bool(name.strip()) and not voted

    if voted:
        st.warning(f"⚠️ 學號 {sid} 已經投過票了，每人只能投一次")
    elif not can_vote:
        st.info("👆 請先填好 第幾組／學號／姓名 才能投票")

    cols = st.columns(4)
    for i, opt in enumerate(options):
        n = counts.get(opt, 0)
        if cols[i].button(
            f"選 {opt}　　{n} 人",
            key=f"v6b_{opt}",
            use_container_width=True,
            disabled=not can_vote,
        ):
            ok, msg = gsheet.add_vote(QKEY, opt, group=group, student_id=sid, name=name)
            if ok:
                st.toast(f"✅ {name} 投了「{opt}」")
            else:
                st.error(msg)
            st.rerun()

    status_text = (
        "🟢 已連結 Google Sheet · 全班即時同步"
        if online
        else "🟡 未連結 Google Sheet（fallback：本機 session）"
    )
    st.markdown(
        f"<div style='text-align:center;color:#666;font-size:14px;margin-top:8px'>{status_text}</div>",
        unsafe_allow_html=True,
    )


def s7():  # 第三階段：相關介紹引言
    st.markdown('<h1 class="st">🔗 兩個變數之間，藏著什麼故事？</h1>', unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="line-height:2.0">
<br>
🧠 &nbsp;前面我們<b>一次只看一個科目</b>——平均、中位數、四分位數。<br><br>

有沒有聽過一個迷思：<br>
　　💭 <span style="color:#ffcc00">「<b>國文好的人，數學比較差</b>」</span><br>
　　💭 <span style="color:#ffcc00">「<b>會計好的人都比較細心</b>」</span><br><br>

這些問題在問的，其實都是<b>兩個變數之間的關係</b>——<br>
　　數學上叫它「<span style="color:#ff4b4b;font-weight:bold">相關性（Correlation）</span>」。<br><br>

📌 &nbsp;這節課，我們要：<br>
　　1. 學會用一個叫做「<b>相關係數 r</b>」的數字描述兩科的關係<br>
　　2. 用三屆學長姐的真實成績<b>實際算一次</b><br>
　　3. 揭曉「文組數學差」這個傳說，到底是真是假！
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
📐 &nbsp;<b>相關係數 r</b>（Pearson）是介於 <b>−1 到 +1</b> 的數字
</div>""", unsafe_allow_html=True)

    def _r_demo_fig(kind: str):
        rng = np.random.default_rng(42)
        _x = np.linspace(0, 10, 11)
        if kind == "+1":
            _y, color = _x, "#22c55e"
        elif kind == "-1":
            _y, color = 10 - _x, "#ff4b4b"
        else:  # 0
            _y, color = rng.uniform(0, 10, len(_x)), "#888"
        _fig = px.scatter(x=_x, y=_y, labels={"x": "x", "y": "y"})
        _fig.update_traces(marker=dict(size=14, color=color))
        if kind in ("+1", "-1"):
            x0, y0, x1, y1 = (0, 0, 10, 10) if kind == "+1" else (0, 10, 10, 0)
            _fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                           line=dict(color=color, width=3, dash="dot"))
        _fig.update_layout(height=320, font_size=16,
                           margin=dict(l=40, r=20, t=20, b=40),
                           xaxis=dict(range=[-1, 11]), yaxis=dict(range=[-1, 11]))
        return _fig

    st.markdown("""
<div class="big">
&nbsp;&nbsp;&nbsp;&nbsp;r = +1　完美正相關（所有點在一條斜向上直線）
</div>""", unsafe_allow_html=True)
    with st.expander("👀 點我看 r = +1 長什麼樣子"):
        st.plotly_chart(_r_demo_fig("+1"), use_container_width=True)
        st.caption("所有點精準落在同一條斜向上的直線上 → r = +1（x 越大，y 等比例越大）")

    st.markdown("""
<div class="big">
&nbsp;&nbsp;&nbsp;&nbsp;r =  0　無線性相關
</div>""", unsafe_allow_html=True)
    with st.expander("👀 點我看 r = 0 長什麼樣子"):
        st.plotly_chart(_r_demo_fig("0"), use_container_width=True)
        st.caption("點散得到處都是、看不出方向 → r ≈ 0（x 跟 y 沒有線性關係）")

    st.markdown("""
<div class="big">
&nbsp;&nbsp;&nbsp;&nbsp;r = −1　完美負相關（所有點在一條斜向下直線）
</div>""", unsafe_allow_html=True)
    with st.expander("👀 點我看 r = −1 長什麼樣子"):
        st.plotly_chart(_r_demo_fig("-1"), use_container_width=True)
        st.caption("所有點精準落在同一條斜向下的直線上 → r = −1（x 越大，y 等比例越小）")

    st.markdown("""
<div class="big">
<br>
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

    tab_f, tab_viz, tab_sd, tab_calc, tab_reveal = st.tabs(
        ["📐 公式", "🎨 圖解（四象限）", "📏 算標準差", "🔢 代入計算（國文 vs 數學）", "✅ 揭曉趨勢線"]
    )

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

    with tab_viz:
        st.markdown("#### 把每個學生的 (xi − x̄)(yi − ȳ) 畫出來")
        st.caption(
            "把每個點和『平均線』比較：右上＋左下 → 正貢獻（綠）；左上＋右下 → 負貢獻（紅）。"
            "綠多紅少 → r 為正；紅多綠少 → r 為負。"
        )
        c_opt1, c_opt2, c_opt3 = st.columns([2, 2, 2])
        show = c_opt1.radio("顯示哪些點？", ["全部", "只看正貢獻", "只看負貢獻"],
                            horizontal=True, key="s8b_viz_show")
        show_rect = c_opt2.checkbox("顯示貢獻矩形（面積 ∝ |貢獻|）", value=False,
                                    key="s8b_viz_rect")
        sample_n = c_opt3.slider("抽樣顯示（避免過密）", 30, len(sub), 150, step=10,
                                 key="s8b_viz_n")

        viz = sub.sample(int(sample_n), random_state=1).copy()
        viz["dx"] = viz["統測_國文分數"] - x_bar
        viz["dy"] = viz["統測_數學B分數"] - y_bar
        viz["貢獻"] = viz["dx"] * viz["dy"]
        viz["sign"] = np.where(viz["貢獻"] >= 0, "正貢獻 (右上＋左下)", "負貢獻 (左上＋右下)")
        if show == "只看正貢獻":
            viz = viz[viz["貢獻"] >= 0]
        elif show == "只看負貢獻":
            viz = viz[viz["貢獻"] < 0]

        fig_v = px.scatter(
            viz, x="統測_國文分數", y="統測_數學B分數", color="sign",
            color_discrete_map={"正貢獻 (右上＋左下)": "#22c55e",
                                "負貢獻 (左上＋右下)": "#ff4b4b"},
            labels={"統測_國文分數": "國文 (x)", "統測_數學B分數": "數學 (y)"},
            opacity=0.7,
        )
        fig_v.update_traces(marker=dict(size=9))
        fig_v.add_vline(x=x_bar, line_dash="dash", line_color="#888",
                        annotation_text=f"x̄ = {x_bar:.1f}", annotation_position="top")
        fig_v.add_hline(y=y_bar, line_dash="dash", line_color="#888",
                        annotation_text=f"ȳ = {y_bar:.1f}", annotation_position="right")
        if show_rect:
            for _, row in viz.iterrows():
                rcolor = "#22c55e" if row["貢獻"] >= 0 else "#ff4b4b"
                fig_v.add_shape(type="rect", x0=x_bar, y0=y_bar,
                                x1=row["統測_國文分數"], y1=row["統測_數學B分數"],
                                line=dict(color=rcolor, width=0),
                                fillcolor=rcolor, opacity=0.06, layer="below")
        fig_v.update_layout(font_size=16, height=480, legend_title_text="")
        st.plotly_chart(fig_v, use_container_width=True)

        n_pos = int((sub["統測_國文分數"] - x_bar).mul(sub["統測_數學B分數"] - y_bar).ge(0).sum())
        n_neg = len(sub) - n_pos
        sum_xy = float(((sub["統測_國文分數"] - x_bar) * (sub["統測_數學B分數"] - y_bar)).sum())
        denom  = (len(sub) - 1) * sx * sy
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("正貢獻人數", f"{n_pos:,}")
        c2.metric("負貢獻人數", f"{n_neg:,}")
        c3.metric("Σ (xi−x̄)(yi−ȳ)", f"{sum_xy:,.0f}")
        c4.metric("r", f"{sum_xy/denom:.3f}")
        st.caption(
            "📌 r 的正負完全由分子決定：把所有 (xi−x̄)(yi−ȳ) 加總；"
            "分母 (n−1)·sx·sy 只是把它縮放到 [−1, +1]。"
        )

    with tab_sd:
        st.markdown("#### 標準差是「每個點到平均的距離」的綜合度量")
        st.caption(
            "和上一頁同一張四象限圖。每位學生離平均線的『水平距離』就是 (xi − x̄)、"
            "『垂直距離』就是 (yi − ȳ)。把這些距離平方後加總平均再開根號,"
            "就是 sx 和 sy。距離大 → 分布散;距離小 → 分布集中。"
        )
        st.latex(r"s_x = \sqrt{\frac{1}{n-1}\sum (x_i - \bar{x})^2}\qquad "
                 r"s_y = \sqrt{\frac{1}{n-1}\sum (y_i - \bar{y})^2}")

        c_o1, c_o2, c_o3, c_o4 = st.columns([3, 2, 2, 1.2])
        which = c_o1.radio("顯示哪個方向的距離?",
                           ["兩個都顯示", "只看 x 距離 (sx)", "只看 y 距離 (sy)"],
                           horizontal=True, key="s8b_sd_which")
        show_sd_box = c_o2.checkbox("顯示 ±1σ 區塊 (黃色)", value=True, key="s8b_sd_box")
        sd_n = c_o3.slider("顯示人數", 30, len(sub), 80, step=10, key="s8b_sd_n")
        if "s8b_sd_seed" not in st.session_state:
            st.session_state.s8b_sd_seed = 2
        if c_o4.button("🎲 換一批", key="s8b_sd_reroll", use_container_width=True):
            st.session_state.s8b_sd_seed += 1

        sd_sample = sub.sample(int(sd_n), random_state=st.session_state.s8b_sd_seed).copy()
        sd_sample["dx"] = sd_sample["統測_國文分數"] - x_bar
        sd_sample["dy"] = sd_sample["統測_數學B分數"] - y_bar

        fig_sd = go.Figure()
        # 散點(灰色,凸顯距離線)
        fig_sd.add_trace(go.Scatter(
            x=sd_sample["統測_國文分數"], y=sd_sample["統測_數學B分數"],
            mode="markers",
            marker=dict(size=9, color="#444", line=dict(width=1, color="#fff")),
            hovertext=[f"國文 {x:.0f},數學 {y:.0f}<br>x 距離 {dx:+.1f},y 距離 {dy:+.1f}"
                       for x, y, dx, dy in zip(sd_sample['統測_國文分數'],
                                                sd_sample['統測_數學B分數'],
                                                sd_sample['dx'], sd_sample['dy'])],
            hoverinfo="text",
            showlegend=False,
        ))
        # 為每個點畫水平 / 垂直距離線
        for _, row in sd_sample.iterrows():
            xi, yi = row["統測_國文分數"], row["統測_數學B分數"]
            cx = "#22c55e" if xi >= x_bar else "#ff4b4b"
            cy = "#22c55e" if yi >= y_bar else "#ff4b4b"
            if which != "只看 y 距離 (sy)":
                fig_sd.add_shape(type="line", x0=x_bar, y0=yi, x1=xi, y1=yi,
                                 line=dict(color=cx, width=1.5))
            if which != "只看 x 距離 (sx)":
                fig_sd.add_shape(type="line", x0=xi, y0=y_bar, x1=xi, y1=yi,
                                 line=dict(color=cy, width=1.5, dash="dot"))
        # 平均十字線
        fig_sd.add_vline(x=x_bar, line_dash="dash", line_color="#888",
                         annotation_text=f"x̄ = {x_bar:.1f}", annotation_position="top")
        fig_sd.add_hline(y=y_bar, line_dash="dash", line_color="#888",
                         annotation_text=f"ȳ = {y_bar:.1f}", annotation_position="right")
        # ±1σ 矩形
        if show_sd_box:
            fig_sd.add_shape(type="rect",
                             x0=x_bar - sx, x1=x_bar + sx,
                             y0=y_bar - sy, y1=y_bar + sy,
                             line=dict(color="#ffcc00", width=2),
                             fillcolor="#ffcc00", opacity=0.10, layer="below")
            fig_sd.add_annotation(x=x_bar + sx, y=y_bar + sy,
                                  text=f"±1σ 區: 寬 {2*sx:.1f} × 高 {2*sy:.1f}",
                                  showarrow=False, font=dict(size=13, color="#b58900"),
                                  xanchor="left", yanchor="bottom")
        fig_sd.update_layout(
            font_size=16, height=520,
            xaxis_title="國文 (x)", yaxis_title="數學 (y)",
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_sd, use_container_width=True)
        st.caption("實線 = x 方向距離 (xi − x̄)　·　虛線 = y 方向距離 (yi − ȳ)　·　"
                   "綠 = 高於平均、紅 = 低於平均")

        # 抽樣 6 筆 對照表
        st.markdown("##### 抽樣 6 筆距離計算")
        show_tbl = sd_sample.head(6)[["統測_國文分數", "統測_數學B分數", "dx", "dy"]].copy()
        show_tbl["(xi−x̄)²"] = (show_tbl["dx"] ** 2).round(1)
        show_tbl["(yi−ȳ)²"] = (show_tbl["dy"] ** 2).round(1)
        show_tbl = show_tbl.rename(columns={
            "統測_國文分數": "國文 xi", "統測_數學B分數": "數學 yi",
            "dx": "xi − x̄", "dy": "yi − ȳ",
        })
        show_tbl[["xi − x̄", "yi − ȳ"]] = show_tbl[["xi − x̄", "yi − ȳ"]].round(1)
        st.dataframe(show_tbl, hide_index=True)

        # 全班最終 sx / sy
        sum_sq_x = float(((sub["統測_國文分數"] - x_bar) ** 2).sum())
        sum_sq_y = float(((sub["統測_數學B分數"] - y_bar) ** 2).sum())
        n_all = len(sub)
        c_x, c_y = st.columns(2)
        with c_x:
            st.markdown("**🟦 國文 sx**")
            a1, a2, a3 = st.columns(3)
            a1.metric("Σ(xi−x̄)²", f"{sum_sq_x:,.0f}")
            a2.metric("÷ (n−1)", f"{sum_sq_x/(n_all-1):,.2f}")
            a3.metric("sx = √…", f"{sx:.2f}")
        with c_y:
            st.markdown("**🟥 數學 sy**")
            b1, b2, b3 = st.columns(3)
            b1.metric("Σ(yi−ȳ)²", f"{sum_sq_y:,.0f}")
            b2.metric("÷ (n−1)", f"{sum_sq_y/(n_all-1):,.2f}")
            b3.metric("sy = √…", f"{sy:.2f}")
        st.info(
            f"💡 國文:平均 {x_bar:.1f},標準差 **{sx:.2f}** → 約 68% 的人落在 "
            f"{x_bar - sx:.0f} ~ {x_bar + sx:.0f} 分　|　"
            f"數學:平均 {y_bar:.1f},標準差 **{sy:.2f}** → 約 {y_bar - sy:.0f} ~ {y_bar + sy:.0f} 分"
        )

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
            st.plotly_chart(fig, use_container_width=True)
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


def s8c():  # 小數字練習：手算 r + 四象限
    st.markdown('<span class="tag">活動 3 · 動手練習</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">✏️ 換你算：5 筆小資料的 r</h2>', unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="line-height:1.8">
真實資料太多，先用 <b>5 筆乾淨數字</b>練手感。<br>
口訣：<b>右上＋左下 = 正貢獻</b>　　<b>左上＋右下 = 負貢獻</b>
</div>""", unsafe_allow_html=True)

    data = pd.DataFrame({"i": [1, 2, 3, 4, 5],
                         "x": [1, 2, 3, 4, 5],
                         "y": [2, 3, 5, 4, 6]})
    x_bar = data["x"].mean()
    y_bar = data["y"].mean()
    sx = data["x"].std()
    sy = data["y"].std()
    data["xi−x̄"] = data["x"] - x_bar
    data["yi−ȳ"] = data["y"] - y_bar
    data["(xi−x̄)(yi−ȳ)"] = data["xi−x̄"] * data["yi−ȳ"]
    sum_xy = data["(xi−x̄)(yi−ȳ)"].sum()
    n = len(data)
    r = sum_xy / ((n - 1) * sx * sy)

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("#### 📋 資料")
        st.dataframe(data[["i", "x", "y"]], hide_index=True, use_container_width=True)

        with st.expander("Step 1：先算 x̄ 和 ȳ"):
            st.latex(rf"\bar{{x}} = \frac{{1+2+3+4+5}}{{5}} = {x_bar:.0f}")
            st.latex(rf"\bar{{y}} = \frac{{2+3+5+4+6}}{{5}} = {y_bar:.0f}")

        with st.expander("Step 2：每筆 (xi−x̄)、(yi−ȳ) 與乘積"):
            st.dataframe(
                data[["i", "x", "y", "xi−x̄", "yi−ȳ", "(xi−x̄)(yi−ȳ)"]],
                hide_index=True, use_container_width=True
            )
            st.latex(rf"\sum (x_i-\bar{{x}})(y_i-\bar{{y}}) = {sum_xy:.0f}")

        with st.expander("Step 3：算 sx, sy"):
            st.latex(rf"s_x = \sqrt{{\tfrac{{(x_i-\bar{{x}})^2}}{{n-1}}}} = {sx:.3f}")
            st.latex(rf"s_y = \sqrt{{\tfrac{{(y_i-\bar{{y}})^2}}{{n-1}}}} = {sy:.3f}")

        with st.expander("Step 4：套公式得 r"):
            st.latex(
                rf"r = \frac{{\sum(x_i-\bar{{x}})(y_i-\bar{{y}})}}{{(n-1)\,s_x\,s_y}}"
                rf" = \frac{{{sum_xy:.0f}}}{{4 \times {sx:.3f} \times {sy:.3f}}}"
                rf" = {r:.3f}"
            )
            st.success(f"答案：**r = {r:.2f}**　→ 強正相關 📈")

    with c_right:
        tab_r, tab_sd_small = st.tabs(["🎨 r 的四象限(貢獻矩形)", "📏 算 sx, sy(距離)"])

        with tab_r:
            viz = data.copy()
            viz["sign"] = np.where(viz["(xi−x̄)(yi−ȳ)"] >= 0,
                                   "正貢獻 (右上＋左下)", "負貢獻 (左上＋右下)")
            fig = px.scatter(
                viz, x="x", y="y", color="sign", text="i",
                color_discrete_map={"正貢獻 (右上＋左下)": "#22c55e",
                                    "負貢獻 (左上＋右下)": "#ff4b4b"},
                labels={"x": "x", "y": "y"},
            )
            fig.update_traces(marker=dict(size=20), textposition="top center",
                              textfont=dict(size=14, color="#fff"))
            fig.add_vline(x=x_bar, line_dash="dash", line_color="#888",
                          annotation_text=f"x̄ = {x_bar:.0f}", annotation_position="top")
            fig.add_hline(y=y_bar, line_dash="dash", line_color="#888",
                          annotation_text=f"ȳ = {y_bar:.0f}", annotation_position="right")
            # 每個點畫貢獻矩形
            for _, row in viz.iterrows():
                rcolor = "#22c55e" if row["(xi−x̄)(yi−ȳ)"] >= 0 else "#ff4b4b"
                fig.add_shape(type="rect", x0=x_bar, y0=y_bar, x1=row["x"], y1=row["y"],
                              line=dict(color=rcolor, width=1),
                              fillcolor=rcolor, opacity=0.15, layer="below")
            fig.update_layout(font_size=16, height=440, legend_title_text="",
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

            n_pos = int((data["(xi−x̄)(yi−ȳ)"] >= 0).sum())
            n_neg = n - n_pos
            c1, c2, c3 = st.columns(3)
            c1.metric("正貢獻", f"{n_pos} 點")
            c2.metric("負貢獻", f"{n_neg} 點")
            c3.metric("Σ", f"{sum_xy:.0f}")
            st.caption("綠色矩形 = 正貢獻;矩形面積 ∝ |(xi−x̄)(yi−ȳ)|。")

        with tab_sd_small:
            fig_sd5 = go.Figure()
            # 散點 + 編號
            fig_sd5.add_trace(go.Scatter(
                x=data["x"], y=data["y"], mode="markers+text",
                marker=dict(size=20, color="#444", line=dict(width=1, color="#fff")),
                text=data["i"], textposition="top center",
                textfont=dict(size=14, color="#fff"),
                hovertext=[f"i={i}: x={x},y={y}<br>x−x̄={x-x_bar:+.1f},y−ȳ={y-y_bar:+.1f}"
                           for i, x, y in zip(data['i'], data['x'], data['y'])],
                hoverinfo="text",
                showlegend=False,
            ))
            # 每個點的 x、y 距離線
            for _, row in data.iterrows():
                xi, yi = row["x"], row["y"]
                cx = "#22c55e" if xi >= x_bar else "#ff4b4b"
                cy = "#22c55e" if yi >= y_bar else "#ff4b4b"
                # x 距離(實線)
                fig_sd5.add_shape(type="line", x0=x_bar, y0=yi, x1=xi, y1=yi,
                                  line=dict(color=cx, width=2))
                # y 距離(虛線)
                fig_sd5.add_shape(type="line", x0=xi, y0=y_bar, x1=xi, y1=yi,
                                  line=dict(color=cy, width=2, dash="dot"))
            # 平均線
            fig_sd5.add_vline(x=x_bar, line_dash="dash", line_color="#888",
                              annotation_text=f"x̄ = {x_bar:.0f}", annotation_position="top")
            fig_sd5.add_hline(y=y_bar, line_dash="dash", line_color="#888",
                              annotation_text=f"ȳ = {y_bar:.0f}", annotation_position="right")
            # ±1σ 矩形
            fig_sd5.add_shape(type="rect",
                              x0=x_bar - sx, x1=x_bar + sx,
                              y0=y_bar - sy, y1=y_bar + sy,
                              line=dict(color="#ffcc00", width=2),
                              fillcolor="#ffcc00", opacity=0.10, layer="below")
            fig_sd5.add_annotation(x=x_bar + sx, y=y_bar + sy,
                                   text=f"±1σ 區: 寬 {2*sx:.2f} × 高 {2*sy:.2f}",
                                   showarrow=False, font=dict(size=12, color="#b58900"),
                                   xanchor="left", yanchor="bottom")
            fig_sd5.update_layout(font_size=16, height=440,
                                  xaxis_title="x", yaxis_title="y",
                                  margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_sd5, use_container_width=True)
            st.caption("實線 = x 方向距離 (xi − x̄)　·　虛線 = y 方向距離 (yi − ȳ)　·　"
                       "綠 = 高於平均、紅 = 低於平均")

            # 計算明細
            sd_tbl = data[["i", "x", "y", "xi−x̄", "yi−ȳ"]].copy()
            sd_tbl["(xi−x̄)²"] = (sd_tbl["xi−x̄"] ** 2).round(2)
            sd_tbl["(yi−ȳ)²"] = (sd_tbl["yi−ȳ"] ** 2).round(2)
            st.dataframe(sd_tbl, hide_index=True, use_container_width=True)
            sum_sq_x = float((data["xi−x̄"] ** 2).sum())
            sum_sq_y = float((data["yi−ȳ"] ** 2).sum())
            c_a, c_b = st.columns(2)
            with c_a:
                st.metric("Σ(xi−x̄)²", f"{sum_sq_x:.0f}")
                st.metric("÷ (n−1) = ÷4", f"{sum_sq_x/4:.2f}")
                st.metric("sx = √…", f"{sx:.3f}")
            with c_b:
                st.metric("Σ(yi−ȳ)²", f"{sum_sq_y:.0f}")
                st.metric("÷ (n−1) = ÷4", f"{sum_sq_y/4:.2f}")
                st.metric("sy = √…", f"{sy:.3f}")


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

    # ── 📤 上傳小組發現 ───────────────────────────────────────
    _persist_voter_state()
    st.markdown("---")
    st.markdown("### 📤 把你們組的發現上傳給全班")

    QKEY_S9 = "group_findings"
    online = gsheet.is_connected()
    group, sid, name = _voter_inputs("s9")

    picked = st.text_input(
        "我們的問題", key="s9_pick",
        placeholder="例:專一和專二,哪科對總分影響更大?",
    )

    can_send = (
        bool(group.strip()) and bool(sid.strip()) and bool(name.strip())
        and bool(str(picked).strip())
    )
    c_btn, c_msg = st.columns([1, 3])
    with c_btn:
        if st.button("📤 上傳問題", key="s9_send", use_container_width=True,
                     disabled=not can_send):
            ok, msg = gsheet.add_finding(
                QKEY_S9, picked_question=str(picked), finding="",
                group=group, student_id=sid, name=name,
            )
            if ok:
                st.toast(f"✅ {name} 上傳成功")
            else:
                st.error(msg)
            st.rerun()
    with c_msg:
        if not can_send:
            st.caption("👆 填好 第幾組／學號／姓名 + 寫出問題 才能上傳")

    rows = gsheet.get_findings(QKEY_S9)
    if rows:
        df_f = pd.DataFrame(rows, columns=["問題", "時間", "組別", "學號", "姓名", "我們的發現"])
        df_f = df_f.iloc[::-1].reset_index(drop=True)  # 最新在前
        st.markdown(f"#### 📋 全班問題一覽({len(df_f)} 筆)")
        st.dataframe(df_f[["組別", "姓名", "問題", "時間"]],
                     hide_index=True, use_container_width=True, height=280)
    else:
        st.info("尚無組別上傳。")

    status_text = (
        "🟢 已連結 Google Sheet · 全班即時同步"
        if online
        else "🟡 未連結 Google Sheet(fallback:本機 session)"
    )
    st.markdown(
        f"<div style='text-align:center;color:#666;font-size:14px;margin-top:8px'>{status_text}</div>",
        unsafe_allow_html=True,
    )


_S10_QUESTIONS = [
    ("第一次模擬考到統測分數趨勢怎麼變？", "折線圖"),
    ("不同科別誰高誰低？", "長條圖"),
    ("整個分布長什麼樣？離群值在哪？", "箱型圖"),
]
_S10_OPTIONS = ["折線圖", "長條圖", "箱型圖"]
_S10_QKEY = "match_chart_pretest"


def s10():  # 第二階段：圖表教學引言 + 連連看前測
    _persist_voter_state()
    st.markdown('<h1 class="st">📊 換個角度看資料</h1>', unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="line-height:2.0">
<br>
🤔 &nbsp;剛剛我們看到的<span style="color:#22c55e;font-weight:bold">那張中位數與平均數的圖</span>，它叫做什麼？<br>
</div>""", unsafe_allow_html=True)

    if st.button("🔍 揭曉答案", key="reveal_s10"):
        st.session_state.revealed_s10 = True
    if st.session_state.get("revealed_s10"):
        st.markdown("""
<div class="big" style="line-height:2.0">
　　答：<b>直方圖（Histogram）</b>——把分數切成一段一段，看每段有幾個人。
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="big" style="line-height:1.8">
<br>
📌 &nbsp;但只有直方圖是不夠的——<b>連連看：哪個問題該配哪種圖？</b>
</div>""", unsafe_allow_html=True)

    group, sid, name = _voter_inputs("s10")
    online = gsheet.is_connected()
    voted = bool(sid) and gsheet.has_voted(_S10_QKEY, sid)

    picks: list[str] = []
    for i, (q, _ans) in enumerate(_S10_QUESTIONS):
        c_q, c_pick = st.columns([3, 2])
        c_q.markdown(
            f"<div style='font-size:22px;line-height:2.4'>{i+1}. {q}</div>",
            unsafe_allow_html=True,
        )
        pick = c_pick.selectbox(
            "配對",
            ["（請選擇）"] + _S10_OPTIONS,
            key=f"s10_pick_{i}",
            label_visibility="collapsed",
            disabled=voted,
        )
        picks.append(pick)

    all_picked = all(p in _S10_OPTIONS for p in picks)
    can_submit = (
        bool(group.strip()) and bool(sid.strip()) and bool(name.strip())
        and all_picked and not voted
    )

    if voted:
        st.warning(f"⚠️ 學號 {sid} 已經提交過了，每人只能交一次")
    elif not (group.strip() and sid.strip() and name.strip()):
        st.info("👆 請先填好 第幾組／學號／姓名")
    elif not all_picked:
        st.info("👆 三題都選好之後才能提交")

    c_submit, c_reveal, c_reset = st.columns([2, 2, 1])
    if c_submit.button("📨 提交配對", key="s10_submit",
                       use_container_width=True, disabled=not can_submit):
        choice_str = "|".join(picks)
        ok, msg = gsheet.add_vote(
            _S10_QKEY, choice_str, group=group, student_id=sid, name=name,
        )
        if ok:
            st.toast(f"✅ {name} 已提交")
        else:
            st.error(msg)
        st.rerun()

    if c_reveal.button("🔍 揭曉答案", key="s10_reveal",
                       use_container_width=True):
        st.session_state.revealed_s10_match = True
    if c_reset.button("♻ 清除", key="s10_reset", use_container_width=True):
        gsheet.reset_votes(_S10_QKEY)
        st.session_state.revealed_s10_match = False
        st.rerun()

    st.markdown(
        f"<div style='text-align:center;color:#666;font-size:14px;margin-top:6px'>"
        f"{'🟢 已連結 Google Sheet · 全班即時同步' if online else '🟡 未連結 Google Sheet（fallback：本機 session）'}"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.get("revealed_s10_match"):
        rows = gsheet.get_rows(_S10_QKEY)
        total = len(rows)
        per_q_correct = [0, 0, 0]
        for r in rows:
            if not r:
                continue
            parts = (r[0] or "").split("|")
            for i, (_q, ans) in enumerate(_S10_QUESTIONS):
                if i < len(parts) and parts[i] == ans:
                    per_q_correct[i] += 1

        st.markdown("#### ✅ 正解")
        for i, (q, ans) in enumerate(_S10_QUESTIONS):
            pct = (per_q_correct[i] / total * 100) if total else 0
            st.markdown(
                f"- **{q}** → <span style='color:#22c55e;font-weight:bold'>{ans}</span>"
                + (f"　<span style='color:#888'>（全班 {per_q_correct[i]}/{total}，{pct:.0f}%）</span>" if total else ""),
                unsafe_allow_html=True,
            )

    st.markdown("""
<div class="big" style="line-height:2.0">
<br>
🎯 &nbsp;這節課要學會一個強大的觀念：<br>
　　<b>箱型圖裡的 Q₁、Q₂、Q₃</b>，其實就是統測公布的<br>
　　<span style="color:#ffcc00;font-weight:bold;font-size:28px">頂標 / 前標 / 均標 / 後標 / 底標！</span>
</div>""", unsafe_allow_html=True)


def s11():  # 折線圖（只看 模1–模5 的趨勢）
    st.markdown('<span class="tag">活動 5 · 資訊老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">📈 折線圖：模考成績有進步嗎？</h2>', unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="line-height:1.8">
<b>折線圖</b>適合用於展示數據隨<b>連續時間</b>或其他<b>連續變數</b>（如距離、溫度）的<b>變化趨勢</b>。
</div>""", unsafe_allow_html=True)
    dept_sel = st.multiselect("科別", DEPT_ORDER, default=DEPT_ORDER, key="s11d")
    stages = [(f"模{i}", f"模{i}_總分數") for i in range(1, 6)]
    rows = [{"科別": d, "考試": stage,
             "平均分數": round(wide[wide["科別"] == d][col].mean(), 1)}
            for d in dept_sel for stage, col in stages
            if pd.notna(wide[wide["科別"] == d][col].mean())]
    fig = px.line(pd.DataFrame(rows), x="考試", y="平均分數", color="科別",
                  markers=True, color_discrete_sequence=DEPT_COLOR,
                  category_orders={"考試": [s for s, _ in stages]})
    fig.update_traces(line_width=4, marker_size=14)
    fig.update_layout(font_size=18, height=460)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("📈 從模1到模5，看起來各科別都穩定成長 — 那正式統測呢？下一頁揭曉。")


def s11c():  # 統測來了（揭曉 模5→統測 落差）
    st.markdown('<span class="tag">活動 5 · 揭曉</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">🎉 統測來了！</h2>', unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="line-height:1.8">
模1→模5 看起來大家都穩定進步⋯⋯<b>那正式統測呢？</b>
</div>""", unsafe_allow_html=True)
    dept_sel = st.multiselect("科別", DEPT_ORDER, default=DEPT_ORDER, key="s11cd")
    stages = [(f"模{i}", f"模{i}_總分數") for i in range(1, 6)] + [("統測", "統測_總分數")]
    rows = [{"科別": d, "考試": stage,
             "平均分數": round(wide[wide["科別"] == d][col].mean(), 1)}
            for d in dept_sel for stage, col in stages
            if pd.notna(wide[wide["科別"] == d][col].mean())]
    df = pd.DataFrame(rows)
    fig = px.line(df, x="考試", y="平均分數", color="科別",
                  markers=True, color_discrete_sequence=DEPT_COLOR,
                  category_orders={"考試": [s for s, _ in stages]})
    fig.update_traces(line_width=4, marker_size=14)
    # 強調 模5 → 統測 的落差
    for d in dept_sel:
        sub = df[df["科別"] == d].set_index("考試")
        if "模5" in sub.index and "統測" in sub.index:
            y5 = sub.loc["模5", "平均分數"]
            yt = sub.loc["統測", "平均分數"]
            color = DEPT_COLOR[DEPT_ORDER.index(d) % len(DEPT_COLOR)]
            fig.add_trace(go.Scatter(
                x=["模5", "統測"], y=[y5, yt],
                mode="lines",
                line=dict(color=color, width=8, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_annotation(
                x="統測", y=yt, text=f"Δ {yt - y5:+.1f}",
                showarrow=False, yshift=20 if yt >= y5 else -20,
                font=dict(color=color, size=15),
            )
    fig.add_annotation(x="統測", y=1.0, yref="paper", yanchor="bottom",
                       text="↓ 正式統測", showarrow=False,
                       font=dict(color="#ffcc00", size=14))
    fig.update_layout(font_size=18, height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 虛線標出的是「模5 → 統測」的落差。**模考分數高，統測未必同樣高**——這正是後面要追的問題。")


def s11b():  # 長條圖
    st.markdown('<span class="tag">活動 5 · 資訊老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">📊 長條圖：各科別 × 各科目</h2>', unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="line-height:1.8">
<b>長條圖</b>主要用於比較不同<b>類別</b>之間的數值大小、呈現<b>離散資料</b>的頻率分佈，或展示在一段時間內的趨勢。
</div>""", unsafe_allow_html=True)
    rows2 = [{"科別":d,"科目":s,"平均分數":round(wide[wide["科別"]==d][col].mean(),1)}
             for s,col in SUBJ.items() for d in DEPT_ORDER
             if pd.notna(wide[wide["科別"]==d][col].mean())]
    fig = px.bar(pd.DataFrame(rows2), x="科目", y="平均分數",
                 color="科別", barmode="group", text="平均分數",
                 color_discrete_sequence=DEPT_COLOR)
    fig.update_traces(textposition="outside", textfont_size=14)
    fig.update_layout(font_size=18, height=460, yaxis_range=[0,115])
    st.plotly_chart(fig, use_container_width=True)


def s12():  # 箱型圖
    _persist_voter_state()
    st.markdown('<span class="tag">活動 5 · 重頭戲</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">📦 箱型圖：一張圖看完所有分布</h2>', unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="line-height:1.8">
<b>箱型圖</b>用<b>五個數字</b>（最小、Q₁、中位數、Q₃、最大）一次顯示資料的<b>分布範圍</b>與<b>離群值</b>，最適合在不同類別之間比較分散程度。<br>
<span style="font-size:18px;color:#bbb">📌 離群值 = 落在 <b>Q₁ − 1.5×IQR</b> 以下 或 <b>Q₃ + 1.5×IQR</b> 以上的點（IQR = Q₃ − Q₁）</span>
</div>""", unsafe_allow_html=True)
    subj_sel = st.multiselect("選科目", list(SUBJ.keys()), default=list(SUBJ.keys()), key="s12s")
    rows = [{"科別":r["科別"],"科目":s,"分數":r[col]}
            for s in subj_sel for col in [SUBJ[s]]
            for _,r in wide[["科別",col]].dropna().iterrows()]
    fig = px.box(pd.DataFrame(rows), x="科目", y="分數", color="科別",
                 points="outliers", color_discrete_sequence=DEPT_COLOR,
                 category_orders={"科目":list(SUBJ.keys()),"科別":DEPT_ORDER})
    fig.update_layout(font_size=18, height=560)
    st.plotly_chart(fig, use_container_width=True)

    # ── 學生分析輸入（寫入 Google Sheet） ─────────────────────────────
    st.markdown("---")
    st.markdown("### ✍️ 你的觀察")
    st.caption("從這張箱型圖看出哪些事實？例：「商經科英文中位數比國貿科高」「數學的離群值最多、分布最寬」")

    QKEY = "boxplot_analysis"
    online = gsheet.is_connected()
    group, sid, name = _voter_inputs("s12")
    analysis = st.text_area(
        "請寫下你看到的客觀觀察（兩三句話即可）",
        key="s12_analysis_text",
        height=120,
        placeholder="例：商經科的英文 Q₃ 比國貿科高 5 分；數學的箱體最高最寬，代表平均高、但同學之間落差也最大。",
    )

    submitted = bool(sid) and gsheet.has_voted(QKEY, sid)
    can_submit = (
        bool(group.strip()) and bool(sid.strip()) and bool(name.strip())
        and bool(analysis.strip()) and not submitted
    )
    if submitted:
        st.warning(f"⚠️ 學號 {sid} 已經提交過了，每人只能交一次")
    elif not can_submit:
        st.info("👆 請填好 第幾組／學號／姓名 + 觀察內容才能提交")

    c_submit, c_reset = st.columns([4, 1])
    if c_submit.button("📤 提交觀察", key="s12_submit",
                       use_container_width=True, disabled=not can_submit,
                       type="primary"):
        ok, msg = gsheet.add_vote(
            QKEY, analysis.strip(), group=group, student_id=sid, name=name,
        )
        if ok:
            st.toast(f"✅ {name} 已提交")
        else:
            st.error(msg)
        st.rerun()
    if c_reset.button("♻ 清除", key="s12_reset", use_container_width=True):
        gsheet.reset_votes(QKEY)
        st.rerun()

    st.markdown(
        f"<div style='text-align:center;color:#666;font-size:14px;margin-top:6px'>"
        f"{'🟢 已連結 Google Sheet · 全班即時同步' if online else '🟡 未連結 Google Sheet（fallback：本機 session）'}"
        "</div>",
        unsafe_allow_html=True,
    )

    # 老師可即時瀏覽全班觀察
    rows_in = gsheet.get_rows(QKEY)
    if rows_in:
        with st.expander(f"👀 全班已交 {len(rows_in)} 份觀察（老師檢視用）", expanded=False):
            for r in rows_in:
                txt = r[0] if len(r) > 0 else ""
                grp = r[2] if len(r) > 2 else ""
                nm  = r[4] if len(r) > 4 else ""
                st.markdown(f"- **第 {grp} 組 · {nm}**：{txt}")


# 四技二專統測各科官方三標(資料來源:教育部技專校院招生委員會)
# 欄位:(到考人數, 平均分數, 前標, 均標, 後標, 群類標示)
# 注意:112 學年度官方未公布均標,以 None 表示。
_TWE_OFFICIAL = {
    114: {
        "國文": (62766, 50.56, 62, 50, 40, "共同科目"),
        "英文": (62655, 45.88, 64, 41, 26, "共同科目"),
        "數學": (33543, 40.00, 52, 36, 24, "共同科目(數學 B)"),
        "專一": (16107, 56.43, 70, 56, 42, "專業科目(一) · 商管外語群"),
        "專二": (13713, 49.31, 64, 46, 32, "專業科目(二) · 商業與管理群"),
    },
    113: {
        "國文": (65514, 54.83, 68, 56, 42, "共同科目"),
        "英文": (65379, 46.84, 67, 41, 26, "共同科目"),
        "數學": (35516, 46.49, 60, 44, 32, "共同科目(數學 B)"),
        "專一": (17130, 59.09, 70, 60, 50, "專業科目(一) · 商管外語群"),
        "專二": (14380, 43.52, 58, 38, 28, "專業科目(二) · 商業與管理群"),
    },
    112: {
        "國文": (70201, 51.99, 64, None, 40, "共同科目"),
        "英文": (70010, 43.50, 59, None, 25.5, "共同科目"),
        "數學": (38304, 39.58, 48, None, 28, "共同科目(數學 B)"),
        "專一": (18369, 51.94, 64, None, 38, "專業科目(一) · 商管外語群"),
        "專二": (15141, 41.91, 54, None, 30, "專業科目(二) · 商業與管理群"),
    },
}
# 向後相容:許多地方仍引用 _TWE_114_THREE
_TWE_114_THREE = _TWE_OFFICIAL[114]


def s13():  # Q1/Q2/Q3
    st.markdown('<span class="tag">活動 6 · 數學老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">箱型圖裡的數學 = 統測三標</h2>', unsafe_allow_html=True)
    st.caption("💡 統測每年公布「前標／均標／後標」三標，其實就是 Q₃／Q₂／Q₁——箱型圖的箱體！"
               "（學測才公布五標，多了頂標、底標）")
    st.markdown(
        '🔍 [Google 搜尋:統測 前均後標](https://www.google.com/search?q=%E7%B5%B1%E6%B8%AC+%E5%89%8D%E5%9D%87%E5%BE%8C%E6%A8%99)',
        unsafe_allow_html=True,
    )

    subj = st.selectbox("選一科來解說", list(SUBJ.keys()), key="s13s")
    data = wide[SUBJ[subj]].dropna()
    p12, q1, q2, q3, p88 = data.quantile([.12, .25, .5, .75, .88])
    iqr = q3 - q1
    n_out = int(((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum())

    c_chart, c_stats = st.columns([3, 1])
    with c_chart:
        fig = px.box(wide, y=SUBJ[subj], points="outliers",
                     labels={SUBJ[subj]: f"{subj} 分數"},
                     color_discrete_sequence=["#ff4b4b"])
        # 五個關鍵百分位數對應到統測五標
        marks = [
            (p88, f"頂標 ≈ {p88:.1f}（前 12%）", "#ffcc00"),
            (q3,  f"前標 = Q₃ = {q3:.1f}（前 25%）", "#aaa"),
            (q2,  f"均標 = Q₂ = {q2:.1f}（中位數）", "#aaa"),
            (q1,  f"後標 = Q₁ = {q1:.1f}（後 25%）", "#aaa"),
            (p12, f"底標 ≈ {p12:.1f}（後 12%）", "#ffcc00"),
        ]
        for val, _label, color in marks:
            fig.add_hline(y=val, line_dash="dot", line_color=color, line_width=2)
        # 把標註放在圖外右側，依 y 值排列彼此不重疊
        for val, label, color in marks:
            fig.add_annotation(
                x=1.02, xref="paper", xanchor="left",
                y=val, yref="y",
                text=label, showarrow=False,
                font=dict(size=14, color=color),
            )
        fig.update_layout(font_size=18, height=540,
                          margin=dict(l=60, r=200, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)
    with c_stats:
        st.markdown(f"""
<div class="big">
<span style="color:#ffcc00"><b>頂標</b></span>（前 12%）<br>{p88:.1f}<br><br>
<b>前標</b> = Q₃<br>{q3:.1f}<br><br>
<b>均標</b> = Q₂（中位數）<br>{q2:.1f}<br><br>
<b>後標</b> = Q₁<br>{q1:.1f}<br><br>
<span style="color:#ffcc00"><b>底標</b></span>（後 12%）<br>{p12:.1f}<br><br>
<b>IQR</b>（前–後標）<br>{iqr:.1f}<br><br>
<span style="color:#ff4b4b">離群值</span><br>{n_out} 人
</div>""", unsafe_allow_html=True)

    st.info("📌 **統測三標的數學定義**：把全部考生分數**由高到低排序**，"
            "**前標**＝前 25%（= Q₃）、**均標**＝前 50%（= 中位數 Q₂）、**後標**＝前 75%（= Q₁）。"
            "所以箱型圖的箱體（Q₁ 到 Q₃）就是「**後標到前標**」這 50% 中間考生的範圍！"
            "（圖上多畫的頂標/底標屬於學測系統，統測不公布）")

    # ── 全國統測官方三標 vs 本校(可切換年度) ─────────────────────────────
    st.markdown("---")
    cmp_subjects = [s for s in SUBJ.keys() if any(s in _TWE_OFFICIAL[y] for y in _TWE_OFFICIAL)]
    if cmp_subjects:
        c_pick1, c_pick2, c_pick3 = st.columns([1, 1, 1])
        with c_pick1:
            default_idx = cmp_subjects.index(subj) if subj in cmp_subjects else 0
            subj_cmp = st.selectbox(
                "下半段比對科目(可獨立切換)",
                cmp_subjects,
                index=default_idx,
                key="s13s_cmp",
            )
        with c_pick2:
            year_options = ["全部三屆"]
            if "畢業年度" in wide.columns:
                year_options += [f"{int(y)} 學年度" for y in sorted(wide["畢業年度"].dropna().unique())]
            year_pick = st.selectbox("年度", year_options, key="s13s_year")
        with c_pick3:
            mode = st.selectbox(
                "缺考處理",
                ["只算到考(預設)", "把缺考也納入(NaN→0)"],
                key="s13s_absent",
            )
        include_absent = mode.startswith("把缺考")

        def _slice(year_val):
            if year_val is None:
                base = wide[SUBJ[subj_cmp]]
            else:
                base = wide.loc[wide["畢業年度"] == year_val, SUBJ[subj_cmp]]
            return base.fillna(0) if include_absent else base.dropna()

        if year_pick == "全部三屆" or "畢業年度" not in wide.columns:
            data_cmp = _slice(None)
            year_label = "全部三屆"
            nat_year = 114  # 預設用最新一屆作為全國基準
        else:
            y = int(year_pick.split()[0])
            data_cmp = _slice(y)
            year_label = f"{y} 學年度畢業"
            nat_year = y
        if include_absent:
            year_label += " · 含缺考"

        q1_c, q2_c, q3_c = data_cmp.quantile([.25, .5, .75])

        # 取對應屆別的全國資料(若該屆無此科,退回 114)
        if subj_cmp in _TWE_OFFICIAL.get(nat_year, {}):
            n, mean, top, mid, bot, group_label = _TWE_OFFICIAL[nat_year][subj_cmp]
            nat_year_used = nat_year
        else:
            n, mean, top, mid, bot, group_label = _TWE_OFFICIAL[114][subj_cmp]
            nat_year_used = 114

        mid_str = f"{mid}" if mid is not None else "—(未公布)"
        st.markdown(f"#### 📊 全國統測 {nat_year_used} 學年度 vs 本校 · {subj_cmp}({year_label})")
        st.caption(f"全國資料:{group_label} · 到考 {n:,} 人 · 平均 {mean:.2f}"
                   + ("　· 112 學年度官方未公布均標" if mid is None else ""))
        c_nat, c_cls = st.columns(2)
        c_nat.markdown(f"""
<div class="big" style="line-height:1.6">
<b>🌐 全國 {nat_year_used} 學年度</b><br>
前標:<b>{top}</b>　·　均標:<b>{mid_str}</b>　·　後標:<b>{bot}</b>
</div>""", unsafe_allow_html=True)
        c_cls.markdown(f"""
<div class="big" style="line-height:1.6">
<b>🏫 本校 · {year_label}</b>(n = {len(data_cmp)},平均 {data_cmp.mean():.2f})<br>
前標 Q₃:<b>{q3_c:.1f}</b>　·　均標 Q₂:<b>{q2_c:.1f}</b>　·　後標 Q₁:<b>{q1_c:.1f}</b>
</div>""", unsafe_allow_html=True)

        # ── 客觀對照:本校 − 全國 ──────────────────────────────────
        nat_iqr = top - bot
        cls_iqr = q3_c - q1_c
        d_top, d_bot = q3_c - top, q1_c - bot
        d_iqr = cls_iqr - nat_iqr
        d_mean = data_cmp.mean() - mean
        spread_word = (
            "更集中(IQR 較窄)" if d_iqr < 0
            else "更分散(IQR 較寬)" if d_iqr > 0
            else "與全國一致"
        )
        mid_line = (
            f"- 均標 Q₂:{q2_c:.1f} − {mid} = **{q2_c - mid:+.1f}** 分\n"
            if mid is not None
            else f"- 均標 Q₂:{q2_c:.1f}(全國 {nat_year_used} 未公布均標,無法比對)\n"
        )
        st.info(
            f"**📌 客觀對照(本校 {year_label} − 全國 {nat_year_used},正值＝本校較高)**\n\n"
            f"- 前標 Q₃:{q3_c:.1f} − {top} = **{d_top:+.1f}** 分\n"
            f"{mid_line}"
            f"- 後標 Q₁:{q1_c:.1f} − {bot} = **{d_bot:+.1f}** 分\n"
            f"- 平均:{data_cmp.mean():.2f} − {mean:.2f} = **{d_mean:+.2f}** 分\n"
            f"- 分布寬度(IQR):本校 {cls_iqr:.1f} vs 全國 {nat_iqr} → 本校分布{spread_word}"
        )



def s14():  # EDA 總結
    st.markdown("""
<div class="full center">
<span class="tag">活動 7 · 雙師總結</span><br><br>
<h1 class="st">「在下結論之前<br>先讓資料自己說話」</h1>
<div class="big" style="color:#888;margin-top:40px">
確認資料正確 ✓ &nbsp;&nbsp; 找出潛在趨勢 ✓ &nbsp;&nbsp; 提出值得深入的問題 ✓
</div>
</div>""", unsafe_allow_html=True)


def s15():  # 第四階段：預測引言
    st.markdown('<h1 class="st">🔮 看完過去，現在來預測未來</h1>', unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="line-height:2.0">
<br>
👀 &nbsp;到現在，我們已經會：<br>
　　• 用<b>直方圖、箱型圖</b>看清楚一科的分布<br>
　　• 用<b>相關係數 r</b> 描述兩科之間的關係<br><br>

但這些都只是<b>看見過去</b>。<br><br>

🚀 &nbsp;最關鍵、也是大家最想問的一題：<br>
　　<span style="color:#ff4b4b;font-weight:bold;font-size:32px">
　　「我能不能用模考分數，預測自己的統測？」
　　</span><br><br>

📌 &nbsp;這節課我們要：<br>
　　1. 在散佈圖上找出一條<b>最會預測的直線</b>　y = ax + b<br>
　　2. 學會「最小平方法」——AI 與機器學習的<b>第一個演算法</b><br>
　　3. <b>實機操作</b>：輸入你估計的模考分數，看看你的統測落點<br><br>

⚠️ &nbsp;但記得：<b>預測有極限</b>，最後分數還是看你接下來這幾個月。
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

    tab_step, tab_formula, tab_verify, tab_code = st.tabs(
        ["📋 步驟說明", "📐 公式與代入", "✅ 驗證", "💻 程式碼做法"])

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

        with st.expander("🤔 為什麼要截距 b？不能只有 y = ax 嗎？"):
            st.markdown(f"""
**a 決定「斜的方向」、b 決定「上下平移的位置」——少一個，線就唯一不下來。**

**① 數學角度：直線一定要過重心 (x̄, ȳ)**

我們希望線通過資料中心 (x̄, ȳ) = ({x_bar:.1f}, {y_bar:.1f})。
若只有 y = ax，x = {x_bar:.1f} 時 y = {a:.4f} × {x_bar:.1f} = **{a*x_bar:.1f}**——
但 ȳ = **{y_bar:.1f}**，差了 **{b:.1f}** 分。
這個落差就是截距 b 補上去的高度。

**② 直觀角度：起跑點不一樣**

模考 0 分的人不會在統測也考 0 分；資料告訴我們：x = 0 時 y ≈ **{b:.1f}**。
b 就是線在 y 軸上的「起點」（外推到 x=0 的預測值）。

下圖紅色虛線是「**強制 b = 0**（過原點）」的迴歸線，整條線被卡低，誤差暴增；
綠線是 **正常含 b** 的迴歸線，貼著資料中心走。
""")
            xs_b = np.linspace(0, sub[x_col].max(), 60)
            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(
                x=sub[x_col], y=sub["統測_總分數"],
                mode="markers",
                marker=dict(color="#888", size=6, opacity=0.35),
                name="實際資料",
            ))
            fig_b.add_trace(go.Scatter(
                x=xs_b, y=a * xs_b + b, mode="lines",
                line=dict(color="#22c55e", width=4),
                name=f"含截距：y = {a:.3f}x + {b:.1f}",
            ))
            fig_b.add_trace(go.Scatter(
                x=xs_b, y=a * xs_b, mode="lines",
                line=dict(color="#ff4b4b", width=4, dash="dash"),
                name=f"強制 b=0：y = {a:.3f}x",
            ))
            fig_b.add_trace(go.Scatter(
                x=[x_bar], y=[y_bar], mode="markers+text",
                marker=dict(color="#ffcc00", size=14, symbol="x"),
                text=["(x̄, ȳ) 重心"], textposition="top right",
                textfont=dict(color="#ffcc00", size=13),
                name="資料重心", showlegend=False,
            ))
            fig_b.update_layout(
                height=380, font_size=14,
                xaxis_title=f"模{exam}總分 (x)", yaxis_title="統測總分 (y)",
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
            )
            st.plotly_chart(fig_b, use_container_width=True)
            st.caption("綠線過 (x̄, ȳ) 重心、誤差²總和最小；紅線被原點綁住，整條偏低——這就是 b 存在的理由。")

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

    with tab_code:
        st.markdown("#### 用程式碼算同一件事")
        st.caption(f"用模{exam}總分當 x、統測總分當 y。下面三種寫法結果一致。")

        st.markdown("**① pandas 一行算 r、平均、標準差**")
        st.code(f"""import pandas as pd

df = pd.read_csv("01_wide_scores.csv")
sub = df[["模{exam}_總分數", "統測_總分數"]].dropna()

x_bar = sub["模{exam}_總分數"].mean()    # → {x_bar:.1f}
y_bar = sub["統測_總分數"].mean()        # → {y_bar:.1f}
sx    = sub["模{exam}_總分數"].std()     # → {sx:.2f}
sy    = sub["統測_總分數"].std()         # → {sy:.2f}
r     = sub.corr().iloc[0, 1]           # → {r:.4f}

a = r * sy / sx        # → {a:.4f}
b = y_bar - a * x_bar  # → {b:.1f}
print(f"y = {{a:.4f}} x + {{b:.1f}}")
""", language="python")

        st.markdown("**② numpy：直接套定義公式**")
        st.code(f"""import numpy as np

x = sub["模{exam}_總分數"].to_numpy()
y = sub["統測_總分數"].to_numpy()

a = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
b = y.mean() - a * x.mean()
# a = {a:.4f}, b = {b:.1f}
""", language="python")

        st.markdown("**③ scikit-learn：機器學習函式庫**")
        st.code(f"""from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(sub[["模{exam}_總分數"]], sub["統測_總分數"])
a = model.coef_[0]      # → {a:.4f}
b = model.intercept_    # → {b:.1f}
""", language="python")

        st.info("💡 三種寫法都給同樣的 a 和 b——數學公式跟程式碼算的是同一件事，只是抽象程度不同。")


def s15c():  # 最小平方法 (OLS)
    st.markdown('<span class="tag">活動 8 後置 · 連結 AI</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">🎯 最小平方法：那條線為什麼是「最佳」？</h2>',
                unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="line-height:1.8">
剛剛公式跑出來的 a 和 b 不是隨便挑的——它們讓
<b>所有點到線的距離²總和</b>（SSE）達到<b>最小值</b>。<br><br>
這個演算法叫做：
<span style="color:#ff4b4b;font-weight:bold;font-size:30px">最小平方法（Ordinary Least Squares, OLS）</span>
</div>""", unsafe_allow_html=True)

    sub = wide[["模5_總分數", "統測_總分數"]].dropna()
    sample = sub.sample(min(50, len(sub)), random_state=42)
    _ols = LinearRegression().fit(sub[["模5_總分數"]], sub["統測_總分數"])
    a, b = float(_ols.coef_[0]), float(_ols.intercept_)

    xs = np.linspace(sub["模5_總分數"].min(), sub["模5_總分數"].max(), 60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample["模5_總分數"], y=sample["統測_總分數"],
        mode="markers", marker=dict(color="#4b9eff", size=8, opacity=0.65),
        name="實際資料",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=a * xs + b, mode="lines",
        line=dict(color="#22c55e", width=4),
        name=f"OLS 最佳線 (a={a:.3f}, b={b:.1f})",
    ))
    for _, row in sample.iterrows():
        x_, y_ = row["模5_總分數"], row["統測_總分數"]
        fig.add_shape(type="line", x0=x_, y0=y_, x1=x_, y1=a * x_ + b,
                      line=dict(color="rgba(255,80,80,0.55)", width=1.2, dash="dot"))
    fig.update_layout(
        font_size=15, height=420,
        xaxis_title="模5總分 (x)", yaxis_title="統測總分 (y)",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    sse = float(((sub["統測_總分數"] - (a * sub["模5_總分數"] + b)) ** 2).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("斜率 a", f"{a:.3f}")
    c2.metric("截距 b", f"{b:.1f}")
    c3.metric("Σ(殘差)² = SSE", f"{sse:,.0f}")
    st.caption("紅色虛線 = 殘差（資料點到線的垂直距離）。OLS 找的 a, b 在所有可能組合中讓 Σ(殘差)² 最小。")

    st.info("""
**🤖 為什麼這跟 AI 有關？**

你剛剛做的事，就是**機器學習最基礎的模型**——線性迴歸（Linear Regression）。
最小平方法就是讓電腦「**自動找出最佳參數 a, b**」的方法。

和現在那些動輒上千億參數的大語言模型（ChatGPT、Claude）核心邏輯相同：
　　① 定義一個**誤差函數**（什麼算「錯」）
　　② 想辦法把它**降到最低**（找最佳參數）

你今天動筆算的這條線，就是 AI 一切的起點。
""")


# ── 活動 8：IT 老師 ────────────────────────────────────────────────

def s16():  # 提問 + 初始散佈圖
    st.markdown('<span class="tag">活動 8 · 資訊老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">看得出規律——但要怎麼用一條線描述它？</h2>',
                unsafe_allow_html=True)
    st.markdown("""
<div class="big" style="margin-top:24px;line-height:1.8">
我們已經知道：模考越高，統測也傾向越高（r ≈ 0.87，<b>強正相關</b>）。<br><br>
但「相關係數」只是 −1 到 +1 的<b>一個數字</b>——
能不能<b>畫一條線</b>，讓我們<b>輸入 x（模考）就能算出 y（統測）</b>？<br><br>
先看看資料的分布——
</div>""", unsafe_allow_html=True)
    sub = wide[["模5_總分數","統測_總分數"]].dropna()
    fig = px.scatter(sub, x="模5_總分數", y="統測_總分數", opacity=0.3,
                     labels={"模5_總分數":"第5次模考總分","統測_總分數":"統測總分"},
                     color_discrete_sequence=["#4b9eff"])
    fig.update_layout(font_size=18, height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("👀 如果要用一條直線「貫穿」這群點，你會怎麼畫？下一頁我們來算出最佳的那條。")


# ── 活動 9：數學老師 ───────────────────────────────────────────────

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
        margin = st.slider("門檻 ± 平均分差", 1, 15, 5, step=1, key="s21b_mg",
                            help="以加權平均（0–100）為單位，跨年/跨校公平比較")
        query  = st.text_input("🔍 搜尋學校或科系（可空白）",
                                placeholder="如：政大、會計、資訊管理",
                                key="s21b_q").strip()

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

        def row_w(row):
            try:
                return round(_calc_w(scores, _parse_w(row["各科目加權"])), 2)
            except Exception:
                return None

        # 預先把整個 cat 三年資料都算好，用來判定「114 可以但 113/112 不行」
        cat_data = thresh[thresh["科系分類"] == cat].copy()
        cat_data["你的加權分"] = cat_data.apply(row_w, axis=1)
        cat_data["分差"] = (cat_data["錄取總分數"] - cat_data["你的加權分"]).round(1)

        # 加權平均（0–100）：除以倍率合，跨年跨校才能公平比
        cat_data["倍率合"] = cat_data["各科目加權"].apply(
            lambda f: sum(_parse_w(f).values()) if pd.notna(f) else None
        )
        cat_data["你的加權平均"] = (cat_data["你的加權分"] / cat_data["倍率合"]).round(1)
        cat_data["門檻平均"]    = (cat_data["錄取總分數"] / cat_data["倍率合"]).round(1)
        cat_data["平均分差"]    = (cat_data["門檻平均"] - cat_data["你的加權平均"]).round(1)

        def _fmt_w(formula):
            """『國文*1+英文*1.5+專業(一)*2』→『國×1 英×1.5 專一×2』。"""
            if pd.isna(formula):
                return ""
            short = {"國文":"國","英文":"英","數學":"數","專一":"專一","專二":"專二"}
            ws = _parse_w(formula)
            return " ".join(f"{short.get(k, k)}×{v:g}" for k, v in ws.items())

        cat_data["加權倍率"] = cat_data["各科目加權"].apply(_fmt_w)
        diff_by_year = {
            yr: dict(zip(
                zip(cat_data.loc[cat_data["年度"] == yr, "學校名稱"],
                    cat_data.loc[cat_data["年度"] == yr, "系科組學程名稱"]),
                cat_data.loc[cat_data["年度"] == yr, "平均分差"]
            ))
            for yr in [114, 113, 112]
        }

        def is_new_chance(school, system):
            """114 你的加權平均 ≥ 門檻，但 113 或 112 不到。"""
            d114 = diff_by_year[114].get((school, system))
            if d114 is None or pd.isna(d114) or d114 > 0:
                return False
            d113 = diff_by_year[113].get((school, system))
            d112 = diff_by_year[112].get((school, system))
            fail113 = d113 is not None and not pd.isna(d113) and d113 > 0
            fail112 = d112 is not None and not pd.isna(d112) and d112 > 0
            return fail113 or fail112

        # 先把所有「114 過、113 或 112 不過」的 (學校, 系科) 撈出來
        keys_special = {
            (row["學校名稱"], row["系科組學程名稱"])
            for _, row in cat_data[cat_data["年度"] == 114].iterrows()
            if is_new_chance(row["學校名稱"], row["系科組學程名稱"])
        }

        year_tabs = st.tabs(["📍 114 學年度", "📅 113 學年度", "📅 112 學年度"])
        for tab, yr in zip(year_tabs, [114, 113, 112]):
            with tab:
                th_sub = cat_data[cat_data["年度"] == yr].copy()
                hit = th_sub[th_sub["平均分差"].abs() <= margin].dropna(subset=["你的加權平均"])

                # 113/112：把跨年「新機會」對應列強制納入（即使超出 margin），讓紅底顯示
                if yr != 114 and keys_special:
                    mask_special = th_sub.apply(
                        lambda r: (r["學校名稱"], r["系科組學程名稱"]) in keys_special, axis=1
                    )
                    extra = th_sub[mask_special & ~th_sub.index.isin(hit.index)]
                    hit = pd.concat([hit, extra])

                # 搜尋字串過濾（學校名 + 系科組合搜）
                if query:
                    q_mask = (
                        hit["學校名稱"].str.contains(query, case=False, na=False)
                        | hit["系科組學程名稱"].str.contains(query, case=False, na=False)
                    )
                    hit = hit[q_mask]

                hit_sorted = hit.sort_values("平均分差", key=abs)

                if hit_sorted.empty:
                    msg = f"± {margin} 平均分差內無符合的 {yr} 年 {cat} 志願。"
                    if query:
                        msg = f"關鍵字「{query}」在 {yr} 年 {cat} 無命中。"
                    st.warning(msg)
                    continue

                hint = f"｜搜尋：「{query}」" if query else ""
                st.caption(f"{yr} 年 {cat}｜± {margin} 平均分差{hint}｜共 **{len(hit_sorted)}** 筆")
                show_cols = [c for c in ["學校名稱","系科組學程名稱","學校類型",
                                          "加權倍率",
                                          "你的加權平均","門檻平均","平均分差",
                                          "你的加權分","錄取總分數"]
                             if c in hit_sorted.columns]
                display = hit_sorted[show_cols].head(20)

                bg_color = "#fff3a0" if yr == 114 else "#ffc9c9"
                def _hl(row, _keys=keys_special, _bg=bg_color):
                    if (row["學校名稱"], row["系科組學程名稱"]) in _keys:
                        return [f"background-color: {_bg}; color: #222"] * len(row)
                    return [""] * len(row)

                shown_special = sum(
                    (s, p) in keys_special
                    for s, p in zip(display["學校名稱"], display["系科組學程名稱"])
                )
                styler = display.style.apply(_hl, axis=1).hide(axis="index")
                st.dataframe(styler, height=400, use_container_width=True)

                if shown_special > 0:
                    if yr == 114:
                        st.caption(
                            f"🟡 淡黃色 = **{shown_special}** 筆「今年才上得了」的志願："
                            "114 你過了，但同系所在 113 或 112 的門檻你不到。"
                        )
                    else:
                        st.caption(
                            f"🔴 紅底 = **{shown_special}** 筆同系所「{yr} 年你不到、114 才過」的對照——"
                            "顯示這幾年門檻或加權變化的影響。"
                        )
        st.caption(
            "**平均分差** = 門檻平均 − 你的加權平均（0–100 同一尺度，跨年/跨校公平比較）。"
            "> 0 = 你還差幾分；< 0 = 你已超過門檻。"
        )


def s21b2():  # 落點後反思:優劣勢、心得
    _persist_voter_state()
    st.markdown('<span class="tag">活動 9 · 個人反思</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">📝 落點之後,聊聊你自己</h2>', unsafe_allow_html=True)
    st.caption("看完落點分析,花一點時間整理你對自己的想法。下面三題不一定要長,寫下真實感受就好。")

    QKEY_REF = "self_reflection"
    HEADERS = ["時間", "組別", "學號", "姓名",
               "優勢科目", "對應有利科系",
               "劣勢科目", "對應該避開科系",
               "心情/收穫"]
    online = gsheet.is_connected()
    group, sid, name = _voter_inputs("s21b2")

    st.markdown("##### 💪 你的優勢")
    c1, c2 = st.columns(2)
    adv_subj = c1.text_input("優勢科目", key="s21b2_adv_subj",
                             placeholder="例:英文、專一")
    adv_dept = c2.text_input("對應哪個科系比較有利?", key="s21b2_adv_dept",
                             placeholder="例:應用外語、商管系")

    st.markdown("##### ⚠️ 你的劣勢")
    c3, c4 = st.columns(2)
    weak_subj = c3.text_input("劣勢科目", key="s21b2_weak_subj",
                              placeholder="例:數學B")
    weak_dept = c4.text_input("對應哪個科系該避開?", key="s21b2_weak_dept",
                              placeholder="例:資訊管理、會計類")

    st.markdown("##### 💭 你的心情與收穫")
    feeling = st.text_area(
        "做完這些活動你的心情如何?有沒有更了解數學/統計、或大學相關資訊的用處?",
        key="s21b2_feel", height=130,
        placeholder="例:原本覺得統計只是公式,看到自己的分數變成圖才知道意義……",
    )

    can_send = (
        bool(group.strip()) and bool(sid.strip()) and bool(name.strip())
        and any(x.strip() for x in [adv_subj, adv_dept, weak_subj, weak_dept, feeling])
    )
    c_btn, c_msg = st.columns([1, 3])
    with c_btn:
        if st.button("📤 上傳反思", key="s21b2_send", use_container_width=True,
                     disabled=not can_send):
            from datetime import datetime
            ts = datetime.now().isoformat(timespec="seconds")
            ok, msg = gsheet.add_record(
                QKEY_REF, HEADERS,
                [ts, group, sid, name,
                 adv_subj.strip(), adv_dept.strip(),
                 weak_subj.strip(), weak_dept.strip(),
                 feeling.strip()],
            )
            if ok:
                st.toast(f"✅ {name} 上傳成功")
            else:
                st.error(msg)
            st.rerun()
    with c_msg:
        if not can_send:
            st.caption("👆 填好 第幾組／學號／姓名 + 至少寫一題 才能上傳")

    rows = gsheet.get_records(QKEY_REF, HEADERS)
    if rows:
        df_r = pd.DataFrame(rows, columns=HEADERS)
        df_r = df_r.iloc[::-1].reset_index(drop=True)
        st.markdown(f"#### 📋 全班反思一覽({len(df_r)} 筆)")
        st.dataframe(
            df_r[["組別", "姓名", "優勢科目", "對應有利科系",
                  "劣勢科目", "對應該避開科系", "心情/收穫", "時間"]],
            hide_index=True, use_container_width=True, height=320,
        )
    else:
        st.info("尚無同學上傳。")

    status_text = (
        "🟢 已連結 Google Sheet · 全班即時同步"
        if online
        else "🟡 未連結 Google Sheet(fallback:本機 session)"
    )
    st.markdown(
        f"<div style='text-align:center;color:#666;font-size:14px;margin-top:8px'>{status_text}</div>",
        unsafe_allow_html=True,
    )


def s21c():  # 推甄 vs 分發
    from data_loader import load_thresholds, load_tuijian_stats, load_tuijian_jifen_stats
    st.markdown('<span class="tag">活動 9 · 數學老師</span>', unsafe_allow_html=True)
    st.markdown('<h2 class="st">推甄 vs 分發：誰能上更好的學校？</h2>',
                unsafe_allow_html=True)
    st.caption("分發：你的加權分 ≥ 門檻才能錄取　｜　推甄：看『統測總級分』通過一階，再看備審、面試")

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
        my_jifen = st.number_input("你預估的統測總級分（0–75）", 0, 75,
                                    value=50, step=1, key="s21c_jf",
                                    help="推甄一階看的就是這個，每科 0–15 級分，五科加總")
        cat  = st.radio("科系分類", ["商管類","資訊類","語文類","設計類"], key="s21c_cat")
        margin_rec = st.slider("推甄機會範圍（門檻超過你幾分以內）",
                               10, 100, 50, step=10, key="s21c_mg")

    thresh   = load_thresholds()
    tj_stats = load_tuijian_stats()
    jf_stats = load_tuijian_jifen_stats()

    def _row_w(row):
        try:
            return round(_calc_w(scores, _parse_w(row["各科目加權"])), 2)
        except Exception:
            return None

    def _classify(diff):
        if diff <= 0:
            return "分發可上 ✅"
        elif diff <= margin_rec:
            return "推甄機會 🎯"
        return "差距較大"

    with c_out:
        year_tabs = st.tabs(["📍 114 學年度", "📅 113 學年度", "📅 112 學年度"])
        for tab, year in zip(year_tabs, [114, 113, 112]):
            with tab:
                th_sub = thresh[(thresh["年度"] == year) & (thresh["科系分類"] == cat)].copy()
                th_sub["你的加權分"] = th_sub.apply(_row_w, axis=1)
                th_sub = th_sub.dropna(subset=["你的加權分"])
                th_sub["分差"] = (th_sub["錄取總分數"] - th_sub["你的加權分"]).round(1)
                th_sub["情境"] = th_sub["分差"].apply(_classify)

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

                # 推甄機會名單 + 加權分 + 級分
                df_rec_schools = th_sub[th_sub["情境"]=="推甄機會 🎯"][
                    ["學校名稱","系科組學程名稱","學校類型","你的加權分","錄取總分數","分差"]
                ].copy()

                tj_year = tj_stats[tj_stats["年度"] == year]
                school_agg = (
                    tj_year.groupby("學校名稱")
                    .agg(
                        推甄人數=("推甄人數", "sum"),
                        推甄最低加權分=("推甄最低加權分", "min"),
                        推甄平均加權分=("推甄平均加權分", "mean"),
                    )
                    .reset_index()
                )
                school_agg["推甄平均加權分"] = school_agg["推甄平均加權分"].round(1)
                school_agg["推甄最低加權分"] = school_agg["推甄最低加權分"].round(1)

                jf_year = jf_stats[jf_stats["年度"] == year]
                jf_school_agg = (
                    jf_year.groupby("學校名稱")
                    .agg(
                        推甄最低總級分=("推甄最低總級分", "min"),
                        推甄平均總級分=("推甄平均總級分", "mean"),
                    )
                    .reset_index()
                )
                jf_school_agg["推甄平均總級分"] = jf_school_agg["推甄平均總級分"].round(1)

                evidence = df_rec_schools.merge(school_agg, on="學校名稱", how="left")
                evidence = evidence.merge(jf_school_agg, on="學校名稱", how="left")
                evidence["推甄人數"] = evidence["推甄人數"].fillna(0).astype(int)
                evidence["你的級分vs最低"] = (my_jifen - evidence["推甄最低總級分"]).round(1)

                # 推甄可上 (你的級分 ≥ 推甄最低總級分) AND 分發不到 (分差 > 0)
                evidence["_yellow"] = (
                    evidence["推甄最低總級分"].notna() &
                    (my_jifen >= evidence["推甄最低總級分"]) &
                    (evidence["分差"] > 0)
                )
                # 黃底列優先排前面，其次推甄人數多、門檻高
                evidence = evidence.sort_values(
                    ["_yellow","推甄人數","錄取總分數"],
                    ascending=[False, False, False])

                n_with   = (evidence["推甄人數"] > 0).sum()
                n_yellow = int(evidence["_yellow"].sum())
                msg = (
                    f"{year} 年｜推甄機會 **{n_rec}** 間，其中 **{n_with}** 間有學長姐推甄錄取紀錄。"
                )
                if n_yellow:
                    msg += (
                        f"　🟡 **{n_yellow}** 間：你的 **{my_jifen}** 級分 ≥ 學長姐推甄最低，"
                        "但分發加權分還不到門檻——這就是「推甄翻身」的機會。"
                    )
                else:
                    msg += "　🟡 黃底＝推甄可上但分發不到的學校；目前無此類紀錄。"
                st.caption(msg)

                show_cols = ["學校名稱","系科組學程名稱","學校類型",
                             "你的加權分","錄取總分數","分差",
                             "推甄人數",
                             "推甄最低總級分","推甄平均總級分","你的級分vs最低",
                             "推甄最低加權分","推甄平均加權分"]
                display = evidence[show_cols + ["_yellow"]].head(20).reset_index(drop=True)

                def _hl(row):
                    if row["_yellow"]:
                        return ["background-color: #fff3a0; color: #222"] * len(row)
                    return [""] * len(row)

                styled = (display.style
                          .apply(_hl, axis=1)
                          .format({
                              "推甄最低總級分": "{:.0f}",
                              "推甄平均總級分": "{:.1f}",
                              "你的級分vs最低": "{:+.0f}",
                              "推甄最低加權分": "{:.1f}",
                              "推甄平均加權分": "{:.1f}",
                              "你的加權分":     "{:.1f}",
                              "錄取總分數":     "{:.0f}",
                              "分差":           "{:.1f}",
                          }, na_rep="—")
                          .hide(axis="index")
                          .hide(["_yellow"], axis="columns"))
                st.dataframe(styled, height=420, use_container_width=True)

        st.caption("⚠️ 人數 0 = 本校三屆無紀錄，不代表不可能。"
                   "加權公式以該校最常見公式估算；推甄仍需備審、面試等條件。")


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

    # ── 📎 上傳 Colab 連結 ─────────────────────────────────────
    _persist_voter_state()
    st.markdown("---")
    st.markdown("### 📎 把你的 Colab 連結交上來")
    st.caption("把今天的 Colab 筆記本「共用 → 任何知道連結的人都可檢視」後,把網址貼到下方。")

    QKEY_COLAB = "colab_submissions"
    online = gsheet.is_connected()
    group, sid, name = _voter_inputs("s22")

    colab_url = st.text_input(
        "Colab 連結",
        key="s22_colab_url",
        placeholder="https://colab.research.google.com/drive/...",
    )

    can_send = (
        bool(group.strip()) and bool(sid.strip()) and bool(name.strip())
        and colab_url.strip().startswith("http")
    )
    c_btn, c_msg = st.columns([1, 3])
    with c_btn:
        if st.button("📤 上傳連結", key="s22_send", use_container_width=True,
                     disabled=not can_send):
            ok, msg = gsheet.add_finding(
                QKEY_COLAB, picked_question=colab_url.strip(), finding="",
                group=group, student_id=sid, name=name,
            )
            if ok:
                st.toast(f"✅ {name} 上傳成功")
            else:
                st.error(msg)
            st.rerun()
    with c_msg:
        if not can_send:
            st.caption("👆 填好 第幾組／學號／姓名 + 貼上 Colab 連結(http 開頭)才能上傳")

    rows = gsheet.get_findings(QKEY_COLAB)
    if rows:
        df_c = pd.DataFrame(rows, columns=["Colab 連結", "時間", "組別", "學號", "姓名", "備註"])
        df_c = df_c.iloc[::-1].reset_index(drop=True)
        st.markdown(f"#### 📋 全班繳交一覽({len(df_c)} 筆)")
        st.dataframe(
            df_c[["組別", "姓名", "Colab 連結", "時間"]],
            hide_index=True, use_container_width=True, height=320,
            column_config={
                "Colab 連結": st.column_config.LinkColumn(
                    "Colab 連結", display_text="🔗 開啟"
                ),
            },
        )
    else:
        st.info("尚無同學上傳。")

    status_text = (
        "🟢 已連結 Google Sheet · 全班即時同步"
        if online
        else "🟡 未連結 Google Sheet(fallback:本機 session)"
    )
    st.markdown(
        f"<div style='text-align:center;color:#666;font-size:14px;margin-top:8px'>{status_text}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
SLIDE_FUNCS = [
    s0, s1, s2, s3, s4, s5, s6, s_pay, s6b,              # 0–8
    s10, s11, s11c, s11b, s13, s12, s14,                 # 9–15（原第三階段：圖表教學）
    s7, s8a, s8b, s8c, s9,                               # 16–20（原第二階段：相關介紹）
    s15, s16, s15b, s15c, s21b, s21b2, s21c, s22,        # 21–28
]
N = len(SLIDE_FUNCS)

_persist_voter_state()  # 跨頁鎖住投票身分欄
SLIDE_FUNCS[slide]()
nav_bar(slide, N)
