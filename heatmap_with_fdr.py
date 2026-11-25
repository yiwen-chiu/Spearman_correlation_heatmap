import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# ============================================================
# Step 1. 通用函式：讀取檔案與資料清理
# ============================================================

def read_data_file(path):
    """依照副檔名讀取 Excel, CSV 或 TSV 檔案。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="cp950")
    elif ext == ".tsv":
        try:
            return pd.read_csv(path, sep="\t", encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, sep="\t", encoding="cp950")
    else:
        raise ValueError(f"不支援的檔案格式：{ext}")

def clean_id(value):
    """清理 ID 的前後空白。"""
    if pd.isna(value):
        return np.nan
    return str(value).strip()

def clean_names(names):
    """清理欄位名稱的前後空白。"""
    return [str(name).strip() for name in names]

def shorten_taxonomy(name):
    """只保留分類路徑最後一個層級的名稱。"""
    return str(name).split("|")[-1]

# ============================================================
# Step 2. 讓使用者選擇兩份檔案 (生理資料 & 菌相資料)
# ============================================================

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

messagebox.showinfo(
    "選擇生理資料",
    "請選擇小鼠生理資料檔案。\n\n"
    "預期格式：每列一隻小鼠，每欄一項生理指標。\n"
    "小鼠 ID 應位於第一欄。"
)

phys_path = filedialog.askopenfilename(
    title="選擇小鼠生理資料",
    filetypes=[
        ("支援的資料檔案", "*.xlsx *.xls *.csv *.tsv"),
        ("Excel 檔案", "*.xlsx *.xls"),
        ("CSV 檔案", "*.csv"),
        ("TSV 檔案", "*.tsv"),
        ("所有檔案", "*.*")
    ]
)

if not phys_path:
    root.destroy()
    raise SystemExit("未選擇生理資料，程式結束。")

messagebox.showinfo(
    "選擇菌相資料",
    "請選擇菌相組成資料檔案。\n\n"
    "預期格式：每列一種菌，每欄一隻小鼠。\n"
    "第一欄為菌名，其他欄位名稱為小鼠 ID。"
)

bac_path = filedialog.askopenfilename(
    title="選擇菌相組成資料",
    filetypes=[
        ("支援的資料檔案", "*.xlsx *.xls *.csv *.tsv"),
        ("Excel 檔案", "*.xlsx *.xls"),
        ("CSV 檔案", "*.csv"),
        ("TSV 檔案", "*.tsv"),
        ("所有檔案", "*.*")
    ]
)

if not bac_path:
    root.destroy()
    raise SystemExit("未選擇菌相資料，程式結束。")

# ============================================================
# Step 3. 讀取及檢查資料
# ============================================================

try:
    phys_raw = read_data_file(phys_path)
    bac_raw = read_data_file(bac_path)
except Exception as e:
    messagebox.showerror("讀取錯誤", str(e))
    root.destroy()
    raise SystemExit

phys_raw.columns = clean_names(phys_raw.columns)
bac_raw.columns = clean_names(bac_raw.columns)

if phys_raw.shape[1] < 2:
    messagebox.showerror("資料格式錯誤", "生理資料至少需要 ID 欄與一個生理指標欄。")
    root.destroy()
    raise SystemExit

if bac_raw.shape[1] < 2:
    messagebox.showerror("資料格式錯誤", "菌相資料至少需要菌名欄與一個小鼠樣本欄。")
    root.destroy()
    raise SystemExit

phys_id_col = phys_raw.columns[0]
phys_raw[phys_id_col] = phys_raw[phys_id_col].map(clean_id)

if phys_raw[phys_id_col].isna().any():
    messagebox.showerror("ID 缺失", "生理資料的第一欄有空白 Mouse ID，請先修正。")
    root.destroy()
    raise SystemExit

if phys_raw[phys_id_col].duplicated().any():
    duplicated = phys_raw.loc[phys_raw[phys_id_col].duplicated(keep=False), phys_id_col].tolist()
    messagebox.showerror("重複 ID", f"生理資料有重複 Mouse ID：\n" + ", ".join(map(str, duplicated)))
    root.destroy()
    raise SystemExit

taxon_col = bac_raw.columns[0]
bac_raw[taxon_col] = bac_raw[taxon_col].map(clean_id)

if bac_raw[taxon_col].isna().any():
    messagebox.showerror("菌名缺失", "菌相資料的第一欄有空白菌名，請先修正。")
    root.destroy()
    raise SystemExit

if bac_raw[taxon_col].duplicated().any():
    duplicated = bac_raw.loc[bac_raw[taxon_col].duplicated(keep=False), taxon_col].tolist()
    messagebox.showerror("重複菌名", f"菌相資料有重複菌名：\n" + ", ".join(map(str, duplicated)) + "\n請先確認是否需要合併相同菌名。")
    root.destroy()
    raise SystemExit

bac_sample_cols = list(bac_raw.columns[1:])
cleaned_sample_cols = clean_names(bac_sample_cols)

if len(cleaned_sample_cols) != len(set(cleaned_sample_cols)):
    messagebox.showerror("重複樣本 ID", "菌相資料轉置後會有重複的樣本 ID，請先檢查欄位名稱。")
    root.destroy()
    raise SystemExit

bac_raw.columns = [taxon_col] + cleaned_sample_cols

if any(not sample_id for sample_id in cleaned_sample_cols):
    messagebox.showerror("樣本 ID 缺失", "菌相資料有空白的樣本欄位名稱。")
    root.destroy()
    raise SystemExit

# ============================================================
# Step 4. 轉置菌相資料方向 (轉為 列=小鼠, 欄=菌種)
# ============================================================

bac = bac_raw.set_index(taxon_col).T
bac.index = pd.Index([clean_id(x) for x in bac.index], name="Mouse_ID")

phys = phys_raw.set_index(phys_id_col)
phys.index = pd.Index([clean_id(x) for x in phys.index], name="Mouse_ID")

if bac.index.isna().any() or bac.index.duplicated().any():
    messagebox.showerror("樣本 ID 錯誤", "菌相資料轉置後有缺失或重複的 Mouse ID。")
    root.destroy()
    raise SystemExit

# ============================================================
# Step 5. 檢查與轉換數值欄位
# ============================================================

phys_numeric = phys.copy()
for col in phys_numeric.columns:
    phys_numeric[col] = pd.to_numeric(phys_numeric[col], errors="coerce")

bac_numeric = bac.copy()
for col in bac_numeric.columns:
    bac_numeric[col] = pd.to_numeric(bac_numeric[col], errors="coerce")

phys_candidates = [col for col in phys_numeric.columns if phys_numeric[col].notna().any()]
bac_candidates = [col for col in bac_numeric.columns if bac_numeric[col].notna().any()]

if not phys_candidates:
    messagebox.showerror("沒有可用的生理指標", "生理資料沒有可辨識的數值欄位。")
    root.destroy()
    raise SystemExit

if not bac_candidates:
    messagebox.showerror("沒有可用的菌相數據", "菌相資料沒有可辨識的數值欄位。")
    root.destroy()
    raise SystemExit

# ============================================================
# Step 6. 讓使用者勾選要分析的菌種與生理指標
# ============================================================

def choose_columns(title, candidates):
    result = {"selected": None}
    win = tk.Toplevel(root)
    win.title(title)
    win.geometry("460x540")
    win.attributes("-topmost", True)

    ttk.Label(win, text=title, font=("Arial", 12, "bold")).pack(pady=8)
    ttk.Label(win, text="可按住 Ctrl (Windows) 或 Command (Mac) 多選。").pack(pady=4)

    frame = ttk.Frame(win)
    frame.pack(fill="both", expand=True, padx=12, pady=8)

    scrollbar = ttk.Scrollbar(frame, orient="vertical")
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        frame,
        selectmode=tk.EXTENDED,
        yscrollcommand=scrollbar.set,
        exportselection=False
    )
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    for item in candidates:
        listbox.insert(tk.END, item)

    listbox.select_set(0, tk.END)

    def confirm():
        indices = listbox.curselection()
        selected = [candidates[i] for i in indices]
        if not selected:
            messagebox.showwarning("尚未選擇項目", "請至少選擇一個欄位。", parent=win)
            return
        result["selected"] = selected
        win.destroy()

    def cancel():
        win.destroy()

    buttons = ttk.Frame(win)
    buttons.pack(pady=10)

    ttk.Button(buttons, text="確認", command=confirm).pack(side="left", padx=6)
    ttk.Button(buttons, text="取消", command=cancel).pack(side="left", padx=6)

    win.protocol("WM_DELETE_WINDOW", cancel)
    win.grab_set()
    root.wait_window(win)

    return result["selected"]

selected_pheno = choose_columns("選擇要分析的生理指標", phys_candidates)
if not selected_pheno:
    root.destroy()
    raise SystemExit("未選擇生理指標，程式結束。")

selected_bacteria = choose_columns("選擇要分析的菌種", bac_candidates)
if not selected_bacteria:
    root.destroy()
    raise SystemExit("未選擇菌種，程式結束。")

# ============================================================
# Step 7. 計算相對豐度 (Relative Abundance)
# ============================================================

sample_totals = bac_numeric.sum(axis=1, min_count=1)
valid_totals = sample_totals.where(sample_totals > 0)
bac_numeric = bac_numeric.div(valid_totals, axis=0)

# ============================================================
# Step 8. 依照 Mouse ID 合併資料
# ============================================================

phys_selected = phys_numeric[selected_pheno].copy()
bac_selected = bac_numeric[selected_bacteria].copy()

common_ids = sorted(set(phys_selected.index) & set(bac_selected.index))
if not common_ids:
    messagebox.showerror("無法合併", "兩個檔案沒有相同的 Mouse ID。")
    root.destroy()
    raise SystemExit

merged = phys_selected.join(bac_selected, how="inner", validate="one_to_one")
merged.index.name = "Mouse_ID"
merged_output = merged.reset_index()

# ============================================================
# Step 9. 計算 Spearman Correlation 及 【新增 BH-FDR 多重校正】
# ============================================================

corr_matrix = pd.DataFrame(index=selected_bacteria, columns=selected_pheno, dtype=float)
p_matrix = pd.DataFrame(index=selected_bacteria, columns=selected_pheno, dtype=float)
p_adj_matrix = pd.DataFrame(index=selected_bacteria, columns=selected_pheno, dtype=float)
n_matrix = pd.DataFrame(index=selected_bacteria, columns=selected_pheno, dtype=int)

# 收集全矩陣的所有 p-value 以進行全域 FDR 校正
p_value_records = []

for bac_name in selected_bacteria:
    for pheno_name in selected_pheno:
        pair = merged[[bac_name, pheno_name]].dropna()
        x = pair[bac_name].to_numpy(dtype=float)
        y = pair[pheno_name].to_numpy(dtype=float)

        n_matrix.loc[bac_name, pheno_name] = len(pair)

        if len(pair) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
            corr_matrix.loc[bac_name, pheno_name] = np.nan
            p_matrix.loc[bac_name, pheno_name] = np.nan
            continue

        r, p = spearmanr(x, y)
        corr_matrix.loc[bac_name, pheno_name] = r
        p_matrix.loc[bac_name, pheno_name] = p

        if not np.isnan(p):
            p_value_records.append((bac_name, pheno_name, p))

# 進行 Benjamini-Hochberg (BH) FDR 校正
if p_value_records:
    raw_ps = [rec[2] for rec in p_value_records]
    _, adj_ps, _, _ = multipletests(raw_ps, alpha=0.05, method="fdr_bh")

    for (bac_name, pheno_name, _), p_adj in zip(p_value_records, adj_ps):
        p_adj_matrix.loc[bac_name, pheno_name] = p_adj

# ============================================================
# Step 10. 建立相關圖標註矩陣 (同時考慮 Raw p 與 FDR p_adj)
# ============================================================

annot_matrix = pd.DataFrame("", index=selected_bacteria, columns=selected_pheno)

for bac_name in selected_bacteria:
    for pheno_name in selected_pheno:
        r_val = corr_matrix.loc[bac_name, pheno_name]
        p_val = p_matrix.loc[bac_name, pheno_name]
        p_adj_val = p_adj_matrix.loc[bac_name, pheno_name]

        if pd.isna(r_val) or pd.isna(p_val):
            annot_matrix.loc[bac_name, pheno_name] = "NA"
            continue

        # 標示星號邏輯：
        # 原始 p-value 星號 (*: p<0.05, **: p<0.01, ***: p<0.001)
        # 若通過 FDR 校正 (p_adj < 0.05)，可額外標示 '#' 記號
        stars = ""
        if p_val < 0.001:
            stars = "***"
        elif p_val < 0.01:
            stars = "**"
        elif p_val < 0.05:
            stars = "*"

        fdr_mark = "#" if (not pd.isna(p_adj_val) and p_adj_val < 0.05) else ""

        annot_matrix.loc[bac_name, pheno_name] = f"{r_val:.2f}\n{stars}{fdr_mark}"

# ============================================================
# Step 11. 輸出包含 Spearman r, Raw p, FDR p_adj 及樣本數之 Excel
# ============================================================

output_xlsx = filedialog.asksaveasfilename(
    title="選擇資料與結果的儲存位置",
    defaultextension=".xlsx",
    initialfile="Merged_Mouse_Data.xlsx",
    filetypes=[("Excel 檔案", "*.xlsx")]
)

if not output_xlsx:
    root.destroy()
    raise SystemExit("未選擇輸出位置，程式結束。")

output_dir = os.path.dirname(output_xlsx)
base_name = os.path.splitext(os.path.basename(output_xlsx))[0]

corr_path = os.path.join(output_dir, f"{base_name}_Correlation_Results.xlsx")

with pd.ExcelWriter(corr_path, engine="openpyxl") as writer:
    corr_matrix.to_excel(writer, sheet_name="Spearman_r")
    p_matrix.to_excel(writer, sheet_name="Raw_P_values")
    p_adj_matrix.to_excel(writer, sheet_name="FDR_Adj_P_values")
    n_matrix.to_excel(writer, sheet_name="Sample_Size")

# ============================================================
# Step 12. 繪製 Heatmap (含 Raw p 與 FDR 標記說明)
# ============================================================

sns.set_theme(style="white", font="DejaVu Sans")

fig_width = max(8, len(selected_pheno) * 1.25)
fig_height = max(5, len(selected_bacteria) * 0.65)

plt.figure(figsize=(fig_width, fig_height), dpi=300)

plot_corr = corr_matrix.copy()
plot_corr.index = [shorten_taxonomy(name) for name in plot_corr.index]
plot_corr.columns = [shorten_taxonomy(name) for name in plot_corr.columns]

ax = sns.heatmap(
    plot_corr,
    annot=annot_matrix,
    fmt="",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    mask=plot_corr.isna(),
    cbar_kws={"label": "Spearman correlation (rho)"},
    linewidths=0.8,
    linecolor="white"
)

plt.title(
    "Gut Microbiota vs. Host Metabolic Parameters (with FDR Correction)",
    fontsize=13,
    fontweight="bold",
    pad=12
)

plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10, fontstyle="italic")
plt.tight_layout()

heatmap_path = os.path.join(output_dir, f"{base_name}_Correlation_Heatmap.png")
plt.savefig(heatmap_path, dpi=600, bbox_inches="tight")

print("\n========== 分析完成 ==========")
print(f"相關性分析結果 Excel (含 Raw P 與 FDR P_adj)：{corr_path}")
print(f"相關性熱圖：{heatmap_path}")
print("標註說明：* Raw p < 0.05; ** Raw p < 0.01; *** Raw p < 0.001; # FDR p_adj < 0.05")

root.destroy()
