# 統測資料互動課

資訊 × 數學 共同教學用一頁式 Streamlit 網站。

## 啟動

```bash
# 安裝套件（只需一次）
pip install -r site/requirements.txt

# 啟動網站（在專案根目錄執行）
streamlit run site/app.py
```

瀏覽器會自動開啟 http://localhost:8501

## 六個 Section

| # | 主題 | 數學概念 |
|---|------|---------|
| 0 | 三屆學長姐的真實數字 | 平均、人數 |
| 1 | 班級錄取地圖 | 計數、占比 |
| 2 | 模擬考能預測統測嗎？ | 散佈圖、相關係數、OLS 迴歸、SSE |
| 3 | 落點查詢 | 區間、排序、中位數 |
| 4 | 熱門學校／科系排行 | Top-K、分位數、盒鬚圖 |
| 5 | 幕後程式 | Python → 互動元件 對照 |

## 教學建議（90 分鐘）

- **0–10 分**：開場，介紹資料從 CSV 到網頁的流程
- **10–30 分**：Section 1，敘述統計（數學主講）
- **30–60 分**：Section 2，散佈圖 → 相關 → OLS；學生先「自己畫一條線」，再看 OLS（數學 + 資訊）
- **60–75 分**：Section 3/4，落點查詢與排行（數學）
- **75–90 分**：打開 app.py，讓學生改 3 行看效果（資訊）

## 資料來源

- `exports/analytical/01_wide_scores.csv`：模擬考 + 統測成績（已去個資）
- `exports/analytical/03_admissions.csv`：錄取去向
- `exports/analytical/05_分發門檻_商管群.csv`：112/113/114 年分發門檻
