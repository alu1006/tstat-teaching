"""Google Sheet 連線：用 Service Account 寫入投票結果。

設定步驟（一次性）：
1. 到 https://console.cloud.google.com 建立專案 → 啟用 Google Sheets API + Drive API
2. 建立 Service Account → 下載 JSON 金鑰
3. 把金鑰內容貼到 .streamlit/secrets.toml（請參考 secrets.toml.example）
4. 在 Google Sheets 建一張試算表，把 service account 的 email 加為「編輯者」
5. 把 Sheet ID（網址 /d/ 後那串）貼到 secrets.toml 的 sheet_id

若沒設定，會自動 fallback 到 session_state（不會壞）。

Sheet 欄位格式：choice | timestamp | 組別 | 學號 | 姓名
"""
from __future__ import annotations
import streamlit as st


HEADERS = ["choice", "timestamp", "組別", "學號", "姓名"]


@st.cache_resource
def _client():
    """建立 gspread 連線（cache 避免重複認證）。"""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        info = st.secrets["gcp_service_account"]
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    except Exception:
        return None


def _sheet(worksheet_name: str):
    cli = _client()
    if cli is None:
        return None
    try:
        sid = st.secrets["gsheet"]["sheet_id"]
        sh = cli.open_by_key(sid)
        try:
            ws = sh.worksheet(worksheet_name)
        except Exception:
            ws = sh.add_worksheet(title=worksheet_name, rows=200, cols=10)
            ws.append_row(HEADERS)
            return ws
        # 確保表頭存在
        try:
            first = ws.row_values(1)
            if first[:2] != HEADERS[:2]:
                ws.update("A1:E1", [HEADERS])
        except Exception:
            pass
        return ws
    except Exception:
        return None


def get_votes(question_key: str, options: list[str]) -> dict[str, int]:
    """從 Sheet 讀取目前票數。Fallback：session_state。"""
    fallback_key = f"_votes_{question_key}"
    ws = _sheet(question_key)
    if ws is None:
        return st.session_state.get(fallback_key, {o: 0 for o in options})
    try:
        rows = ws.get_all_values()
        counts = {o: 0 for o in options}
        for r in rows[1:]:
            if r and r[0] in counts:
                counts[r[0]] += 1
        return counts
    except Exception:
        return st.session_state.get(fallback_key, {o: 0 for o in options})


def has_voted(question_key: str, student_id: str) -> bool:
    """檢查該學號是否已投過票。"""
    if not student_id:
        return False
    fallback_key = f"_voters_{question_key}"
    ws = _sheet(question_key)
    if ws is None:
        return student_id in st.session_state.get(fallback_key, set())
    try:
        rows = ws.get_all_values()
        # 學號在第 4 欄（index 3）
        for r in rows[1:]:
            if len(r) >= 4 and r[3].strip() == student_id.strip():
                return True
        return False
    except Exception:
        return False


def add_vote(
    question_key: str,
    choice: str,
    group: str = "",
    student_id: str = "",
    name: str = "",
    allow_duplicate: bool = False,
) -> tuple[bool, str]:
    """寫入一票。回傳 (是否成功, 訊息)。

    預設:該學號已投過時擋下。allow_duplicate=True 時允許重複提交。
    """
    if not allow_duplicate and student_id and has_voted(question_key, student_id):
        return False, f"學號 {student_id} 已經投過票了"

    fallback_key = f"_votes_{question_key}"
    voters_key = f"_voters_{question_key}"
    ws = _sheet(question_key)

    if ws is None:
        # fallback to session state
        if fallback_key not in st.session_state:
            st.session_state[fallback_key] = {}
        st.session_state[fallback_key][choice] = (
            st.session_state[fallback_key].get(choice, 0) + 1
        )
        if voters_key not in st.session_state:
            st.session_state[voters_key] = set()
        if student_id:
            st.session_state[voters_key].add(student_id)
        return True, "已記錄（本機暫存）"

    try:
        from datetime import datetime
        ws.append_row([
            choice,
            datetime.now().isoformat(timespec="seconds"),
            group,
            student_id,
            name,
        ])
        return True, "已寫入 Google Sheet"
    except Exception as e:
        return False, f"寫入失敗：{e}"


def get_rows(question_key: str) -> list[list[str]]:
    """讀取整個工作表的資料列（不含表頭）。讀不到時回 []。"""
    ws = _sheet(question_key)
    if ws is None:
        return []
    try:
        rows = ws.get_all_values()
        return rows[1:] if rows else []
    except Exception:
        return []


def reset_votes(question_key: str):
    """清空一個問題的所有投票。"""
    ws = _sheet(question_key)
    if ws is not None:
        try:
            ws.clear()
            ws.append_row(HEADERS)
            return True
        except Exception:
            pass
    fallback_key = f"_votes_{question_key}"
    voters_key = f"_voters_{question_key}"
    st.session_state[fallback_key] = {}
    st.session_state[voters_key] = set()
    return False


def is_connected() -> bool:
    return _client() is not None


# ── 小組發現:6 欄位、不做唯一性檢查 ─────────────────────────────────
FINDING_HEADERS = ["picked_question", "timestamp", "組別", "學號", "姓名", "我們的發現"]


def _finding_sheet(worksheet_name: str):
    cli = _client()
    if cli is None:
        return None
    try:
        sid = st.secrets["gsheet"]["sheet_id"]
        sh = cli.open_by_key(sid)
        try:
            ws = sh.worksheet(worksheet_name)
        except Exception:
            ws = sh.add_worksheet(title=worksheet_name, rows=200, cols=10)
            ws.append_row(FINDING_HEADERS)
            return ws
        try:
            first = ws.row_values(1)
            if first[:6] != FINDING_HEADERS:
                ws.update("A1:F1", [FINDING_HEADERS])
        except Exception:
            pass
        return ws
    except Exception:
        return None


def add_finding(
    question_key: str,
    picked_question: str,
    finding: str,
    group: str = "",
    student_id: str = "",
    name: str = "",
) -> tuple[bool, str]:
    """寫入一筆小組發現。允許同一人重複提交。"""
    ws = _finding_sheet(question_key)
    fallback_key = f"_findings_{question_key}"
    from datetime import datetime
    ts = datetime.now().isoformat(timespec="seconds")
    row = [picked_question, ts, group, student_id, name, finding]

    if ws is None:
        if fallback_key not in st.session_state:
            st.session_state[fallback_key] = []
        st.session_state[fallback_key].append(row)
        return True, "已記錄(本機暫存)"
    try:
        ws.append_row(row)
        return True, "已寫入 Google Sheet"
    except Exception as e:
        return False, f"寫入失敗:{e}"


def get_findings(question_key: str) -> list[list[str]]:
    ws = _finding_sheet(question_key)
    fallback_key = f"_findings_{question_key}"
    if ws is None:
        return st.session_state.get(fallback_key, [])
    try:
        rows = ws.get_all_values()
        return rows[1:] if rows else []
    except Exception:
        return st.session_state.get(fallback_key, [])


# ── 通用多欄位記錄 ───────────────────────────────────────────────
def _record_sheet(worksheet_name: str, headers: list[str]):
    cli = _client()
    if cli is None:
        return None
    try:
        sid = st.secrets["gsheet"]["sheet_id"]
        sh = cli.open_by_key(sid)
        try:
            ws = sh.worksheet(worksheet_name)
        except Exception:
            ws = sh.add_worksheet(title=worksheet_name, rows=300, cols=max(10, len(headers)))
            ws.append_row(headers)
            return ws
        try:
            first = ws.row_values(1)
            if first[:len(headers)] != headers:
                end_col = chr(ord("A") + len(headers) - 1)
                ws.update(f"A1:{end_col}1", [headers])
        except Exception:
            pass
        return ws
    except Exception:
        return None


def add_record(question_key: str, headers: list[str], values: list) -> tuple[bool, str]:
    """通用:寫入一列任意欄位的紀錄。values 順序需對應 headers。"""
    ws = _record_sheet(question_key, headers)
    fallback_key = f"_records_{question_key}"
    row = [str(v) for v in values]

    if ws is None:
        if fallback_key not in st.session_state:
            st.session_state[fallback_key] = []
        st.session_state[fallback_key].append(row)
        return True, "已記錄(本機暫存)"
    try:
        ws.append_row(row)
        return True, "已寫入 Google Sheet"
    except Exception as e:
        return False, f"寫入失敗:{e}"


def get_records(question_key: str, headers: list[str]) -> list[list[str]]:
    ws = _record_sheet(question_key, headers)
    fallback_key = f"_records_{question_key}"
    if ws is None:
        return st.session_state.get(fallback_key, [])
    try:
        rows = ws.get_all_values()
        return rows[1:] if rows else []
    except Exception:
        return st.session_state.get(fallback_key, [])
