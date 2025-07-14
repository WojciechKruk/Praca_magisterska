import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
#  Parametry “na wierzchu”
# ─────────────────────────────────────────
DEBUG_ALIGN = False          # True ➜ wypisuje przypadki z użyciem luki

BASE = Path(__file__).resolve().parent
OUT  = BASE / "results_analysis"
OUT.mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────
#  Stałe – ścieżki
# ─────────────────────────────────────────
PATHS = {
    "labels" : BASE / "labels.txt",
    "cvx_res": next((BASE/"CV-Xv2").glob("*result*")),
    "cvx_tm" : next((BASE/"CV-Xv2").glob("*time*")),
    "my_res" : next((BASE/"My_modelv2").glob("*result*")),
    "my_tm"  : next((BASE/"My_modelv2").glob("*time*")),
}
for k,p in PATHS.items():
    if p is None or not p.exists():
        raise FileNotFoundError(f"Brak pliku: {k}")

# ─────────────────────────────────────────
#  Reg-exp do plików
# ─────────────────────────────────────────
RE_RES = re.compile(r"(\S+)\s+(\S+)\s+(\S+)")
RE_TM  = re.compile(
    r"Detect:\s*([\d.]+).*?"
    r"Decode[ _]date:\s*([\d.]+).*?"
    r"Decode[ _]code:\s*([\d.]+).*?"
    r"Detect \+ decode:\s*([\d.]+).*?"
    r"Total\s+([\d.]+)", re.I)

# ─────────────────────────────────────────
#  Alphabety
# ─────────────────────────────────────────
ALPH_DATE = set("0123456789.") | {" "}
ALPH_CODE = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:") | {" "}
ALPH_BOTH = ALPH_DATE | ALPH_CODE

# ══════════════════════════════════════════════════════════════
#  0.  FUNKCJE ŁADOWANIA
# ══════════════════════════════════════════════════════════════
def load_results(path: Path) -> dict[str, tuple[str,str]]:
    out={}
    for m in map(RE_RES.match, path.read_text().splitlines()):
        if m: out[m[1]]=(m[2],m[3])
    return out

def load_times(path: Path, system:str) -> pd.DataFrame:
    rows=[]
    for n,line in enumerate(path.read_text().splitlines(),1):
        m=RE_TM.search(line)
        if not m: continue
        rows.append(dict(
            n=n, system=system, file=line.split()[0],
            detect=float(m[1]), decode_date=float(m[2]),
            decode_code=float(m[3]), detect_decode=float(m[4]),
            total=float(m[5])))
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════
#  1.  “SPRYTNE” WYRÓWNANIE PAR ZNAKÓW
# ══════════════════════════════════════════════════════════════
def align_pairs(ref:str, pred:str):
    """
    Inteligentne wyrównanie:
    ● kiedy trafiamy na rozbieżność, wolno pominąć 1 znak
      w ref *lub* 1 znak w pred, ale tylko wtedy, gdy
        – jesteśmy na przedostatnim znaku  **lub**
        – po pominięciu **kolejne 2** znaki się zgadzają.
    Zwraca:   [(ref_ch, pred_ch), …],   used_gap(bool)
    """
    pairs, i, j, used_gap = [], 0, 0, False
    while i < len(ref) or j < len(pred):

        # pred się skończył  →  “spacje” po stronie pred
        if j >= len(pred):
            pairs.append((ref[i], " ")); i += 1; continue
        # ref się skończył  →  “spacje” po stronie ref
        if i >= len(ref):
            pairs.append((" ", pred[j])); j += 1; continue

        if ref[i] == pred[j]:
            pairs.append((ref[i], pred[j])); i += 1; j += 1; continue

        # ­­­— kandydat: pomijamy ref[i] —
        skip_ref_ok = (
            i+1 < len(ref) and ref[i+1]==pred[j] and
            (len(ref)-i-1==1 or (
               i+2 < len(ref) and j+1 < len(pred) and
               ref[i+2]==pred[j+1]))
        )
        # ­­­— kandydat: pomijamy pred[j] —
        skip_pred_ok = (
            j+1 < len(pred) and pred[j+1]==ref[i] and
            (len(pred)-j-1==1 or (
               i+1 < len(ref) and j+2 < len(pred) and
               ref[i+1]==pred[j+2]))
        )

        if skip_ref_ok:
            pairs.append((ref[i], " ")); i += 1; used_gap=True; continue
        if skip_pred_ok:
            pairs.append((ref[i], pred[j])); j += 1; used_gap=True; continue

        # zwykła pomyłka
        pairs.append((ref[i], pred[j])); i += 1; j += 1
    return pairs, used_gap

# ══════════════════════════════════════════════════════════════
#  2.  METRYKI
# ══════════════════════════════════════════════════════════════
def seq_accuracy(ref,pred,idx):
    ok=sum(1 for f,(d,c) in ref.items()
           if f in pred and (d,c)[idx]==pred[f][idx])
    return 100*ok/len(ref)

def char_acc_simple(ref,pred,idx):
    hit=tot=0
    for f,(d,c) in ref.items():
        if f not in pred: continue
        r,p=(d,c)[idx], pred[f][idx]
        hit += sum(a==b for a,b in zip(r,p))
        tot += len(r)
    return 100*hit/tot if tot else 0

def char_acc_gap(ref,pred,idx):
    hit=tot=0
    for f,(d,c) in ref.items():
        if f not in pred: continue
        r, p = (d,c)[idx], pred[f][idx]
        pairs, gap = align_pairs(r, p)

        if DEBUG_ALIGN and gap:
            print(f"{f}: ref='{r}'  pred='{p}'")
            print("     " + ''.join(a for a,_ in pairs))
            print("     " + ''.join(b for _,b in pairs))

        for a,b in pairs:
            if a==" ": continue
            tot += 1
            if a==b: hit += 1
    return 100*hit/tot if tot else 0

# ══════════════════════════════════════════════════════════════
#  3.  CZASY  +  WYKRESY CZASÓW
# ══════════════════════════════════════════════════════════════
def plot_times(time_df: pd.DataFrame, cvx_res: dict, my_res: dict, labels: dict):
    label_map = {
        "detect": "Czas działania modułu detekcji [s]",
        "decode_date": "Czas dekodowania segmentu daty [s]",
        "decode_code": "Czas dekodowania segmentu kodu [s]",
        "detect_decode": "Całkowity czas detekcji i dekodowania [s]",
        "total": "Całkowity czas przetwarzania obrazu [s]"
    }

    for col in label_map:
        plt.figure(figsize=(9, 4))
        for sys, grp in time_df.groupby("system"):
            plt.plot(grp.n, grp[col], label=f"{sys}")
        plt.xlabel("Numer próbki")
        plt.ylabel(label_map[col])
        plt.title(label_map[col])
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"{col}.png", dpi=300)
        plt.close()

    for metric in ["detect", "total"]:
        plt.figure(figsize=(5, 4))
        data = [time_df.query("system=='CV-X'")[metric],
                time_df.query("system=='My_model'")[metric]]
        plt.boxplot(data, tick_labels=["CV-X", "Model własny"], showfliers=False)
        plt.ylabel(label_map[metric])
        plt.title(f"{label_map[metric]}\n– rozkład dla obu systemów")
        plt.tight_layout()
        plt.savefig(OUT / f"{metric}_box.png", dpi=300)
        plt.close()

    def seq_ok(pred):
        return pd.Series([int(f in pred and pred[f] == labels[f]) for f in labels], name="ok")

    df = pd.concat([
        time_df.query("system=='CV-X'").reset_index(drop=True).assign(ok=seq_ok(cvx_res)),
        time_df.query("system=='My_model'").reset_index(drop=True).assign(ok=seq_ok(my_res))
    ], ignore_index=True)

    color_map = {
        ("CV-X", 1): "tab:blue",
        ("CV-X", 0): "#A6C8FF",
        ("My_model", 1): "tab:orange",
        ("My_model", 0): "#FFD8A6",
    }
    marker_map = {1: "o", 0: "x"}

    # plt.figure(figsize=(7, 5))
    # for (sys, ok_val), grp in df.groupby(["system", "ok"]):
    #     plt.scatter(grp.total, grp.n,
    #                 c=color_map[(sys, ok_val)],
    #                 marker=marker_map[ok_val],
    #                 s=35,
    #                 label=f"{sys} – {'poprawnie' if ok_val else 'błędnie'}")

    # Najpierw poprawne odczyty (ok == 1)
    for sys in ["CV-X", "My_model"]:
        grp = df[(df.system == sys) & (df.ok == 1)]
        plt.scatter(grp.total, grp.n,
                    c=color_map[(sys, 1)],
                    marker=marker_map[1],
                    s=35,
                    label=f"{sys} – poprawnie")

    # Potem błędne odczyty (ok == 0) → będą na wierzchu
    for sys in ["CV-X", "My_model"]:
        grp = df[(df.system == sys) & (df.ok == 0)]
        plt.scatter(grp.total, grp.n,
                    c=color_map[(sys, 0)],
                    marker=marker_map[0],
                    s=35,
                    label=f"{sys} – błędnie")

    plt.xlabel("Całkowity czas przetwarzania obrazu [s]")
    plt.ylabel("Numer próbki")
    plt.title("Czas działania systemu wizyjnego w funkcji numeru próbki, z podziałem na poprawne i błędne odczyty", wrap=True)
    plt.gca().invert_yaxis()
    plt.legend(title="Legenda", framealpha=1, edgecolor="#999999")
    plt.tight_layout()
    plt.savefig(OUT / "scatter_total_vs_n.png", dpi=300)
    plt.close()


# ══════════════════════════════════════════════════════════════
#  zbiorczy wykres słupkowy – 6 metryk / system
# ══════════════════════════════════════════════════════════════
def plot_big_bar(seq_df: pd.DataFrame, char_old_df: pd.DataFrame, char_gap_df: pd.DataFrame):
    systems = ["CV-X", "My_model"]
    metrics = [
        ("Dokładność sekwencyjna (data)",  seq_df["date"]),
        ("Dokładność znakowa (data)", char_old_df["date"]),
        ("Gap-aware dokładność znakowa (data)",  char_gap_df["date"]),
        ("Dokładność sekwencyjna (kod)",  seq_df["code"]),
        ("Dokładność znakowa (kod)", char_old_df["code"]),
        ("Gap-aware dokładność znakowa (kod)",  char_gap_df["code"]),
    ]

    bar_w = 0.12
    x_base = np.arange(len(systems))

    plt.figure(figsize=(10, 5))
    for i, (lbl, serie) in enumerate(metrics):
        x = x_base + (i - 2.5) * bar_w
        plt.bar(x, [serie[s] for s in systems], bar_w, label=lbl, alpha=.9)

    plt.xticks(x_base, ["CV-X", "Model własny"])
    plt.ylabel("Dokładność [%]")
    plt.ylim(0, 130)
    plt.title("Porównanie dokładności dla sześciu miar")
    plt.legend(ncol=2, framealpha=.95)
    plt.tight_layout()
    plt.savefig(OUT / "accuracy_all_metrics_bar.png", dpi=300)
    plt.close()


# ══════════════════════════════════════════════════════════════
#  4.  MACIERZE POMYŁEK  (errors only, gap-aware)
# ══════════════════════════════════════════════════════════════
def build_cm(ref, pred, field_idx):
    alphabet = ( sorted(ALPH_DATE) if field_idx==0 else
                 sorted(ALPH_CODE) if field_idx==1 else
                 sorted(ALPH_BOTH) )
    lab2idx={c:i for i,c in enumerate(alphabet)}
    cm=np.zeros((len(alphabet),len(alphabet)),dtype=int)

    for f,(d,c) in ref.items():
        if f not in pred: continue
        r  = (d,c)[field_idx] if field_idx in (0,1) else d+c
        p  = pred[f][field_idx] if field_idx in (0,1) else ''.join(pred[f])

        for a,b in align_pairs(r,p)[0]:
            if a==" ": continue
            if a not in lab2idx or b not in lab2idx: continue
            cm[lab2idx[a], lab2idx[b]] += 1
    return cm, alphabet

def plot_cm(cm,alphabet,title,fname):
    cm_norm = cm / cm.sum(axis=1,keepdims=True).clip(min=1)
    np.fill_diagonal(cm_norm,0)                 # tylko błędy
    cm_norm *= 100

    plt.figure(figsize=(8,8))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=cm_norm.max()+1e-6)
    plt.colorbar(fraction=.046)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(alphabet)), alphabet, rotation=90)
    plt.yticks(range(len(alphabet)), alphabet)
    plt.title(title); plt.tight_layout()
    plt.savefig(OUT/fname, dpi=300); plt.close()

def generate_confusions(labels, cvx_res, my_res):
    systems={"CV-X": cvx_res, "My_model": my_res}
    fields ={ "both":None, "date":0, "code":1 }
    for sys,pred in systems.items():
        for name,idx in fields.items():
            cm,alph = build_cm(labels,pred,idx)
            plot_cm(cm,alph,f"Errors only – {sys} – {name}",
                    f"cm_err_{sys}_{name}.png")


# ══════════════════════════════════════════════════════════════
# 6.  DEBUG – pełna lista pomyłek wybranego znaku
# ══════════════════════════════════════════════════════════════
def list_mistakes_for_char(
        the_char: str,
        labels: dict[str, tuple[str, str]],
        pred: dict[str, tuple[str, str]],
        field: str = "both",  # "date" | "code" | "both"
        system: str = "sys",
) -> Path:
    if field not in {"date", "code", "both"}:
        raise ValueError("field musi być 'date', 'code' lub 'both'")

    rows = []
    fld_idx = 0 if field == "date" else 1 if field == "code" else None

    for f, (d, c) in labels.items():
        if f not in pred: continue
        r = (d, c)[fld_idx] if fld_idx is not None else d + c
        p = pred[f][fld_idx] if fld_idx is not None else ''.join(pred[f])

        for ref_ch, pred_ch in align_pairs(r, p)[0]:
            if ref_ch == " ":  # luka w ref – nie liczymy
                continue
            if ref_ch != pred_ch and (ref_ch == the_char or pred_ch == the_char):
                rows.append((f, r, p))
                break  # wystarczy 1 błąd w linii

    if not rows:
        print(f"⚠️  Brak pomyłek znaku '{the_char}' w {system}/{field}")
        return None

    df = pd.DataFrame(rows, columns=["file", "ref", "pred"])
    out_path = OUT / f"mistakes_{system}_{field}_{the_char}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"🔍  {len(df)} pomyłek znaku '{the_char}' zapisano w {out_path}")
    return out_path

# ══════════════════════════════════════════════════════════════
#  5.  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    # ――― 1. Wczytanie danych ―――
    labels   = load_results(PATHS["labels"])
    cvx_res  = load_results(PATHS["cvx_res"])
    my_res   = load_results(PATHS["my_res"])

    time_df  = pd.concat([
        load_times(PATHS["cvx_tm"], "CV-X"),
        load_times(PATHS["my_tm"],  "My_model")
    ], ignore_index=True)

    # ---- dokładność sekwencji --------------------------------
    seq_res = { s:[seq_accuracy(labels,d,0), seq_accuracy(labels,d,1)]
                for s,d in [("CV-X",cvx_res),("My_model",my_res)] }
    seq_df=pd.DataFrame(seq_res,index=["date","code"]).T.round(2)
    seq_df.to_csv(OUT/"sequence_accuracy.csv")
    print("\n=== dokładność ciągu (%) ===\n",seq_df)

    # ---- dokładność znakowa ----------------------------------
    char_old = {s:[char_acc_simple(labels,d,0), char_acc_simple(labels,d,1)]
                for s,d in [("CV-X",cvx_res),("My_model",my_res)]}
    char_gap = {s:[char_acc_gap(labels,d,0), char_acc_gap(labels,d,1)]
                for s,d in [("CV-X",cvx_res),("My_model",my_res)]}

    old_df=pd.DataFrame(char_old, index=["date","code"]).T.round(2)
    gap_df=pd.DataFrame(char_gap, index=["date","code"]).T.round(2)
    old_df.to_csv(OUT/"char_accuracy_simple.csv")
    gap_df.to_csv(OUT/"char_accuracy_gap.csv")

    print("\n=== dokładność znakowa *stara* (%) ===\n", old_df)
    print("\n=== dokładność znakowa *gap-aware* (%) ===\n", gap_df)

    # wykres słupkowy gap-aware
    plt.figure(figsize=(6,4))
    x=np.arange(2); w=.35
    for i,(sys,row) in enumerate(gap_df.iterrows()):
        plt.bar(x + (i-0.5)*w, row, w, label=sys)
    # plt.xticks(x,["date","code"]); plt.ylim(0,100); plt.ylabel("%")
    plt.title("Gap-aware dokładność znakowa dla segmentów daty i kodu")
    plt.ylim(0, 130)
    plt.ylabel("Dokładność [%]")
    plt.xticks(x, ["Segment daty", "Segment kodu"])
    plt.legend(title="System")
    plt.savefig(OUT/"char_accuracy_gap_bar.png", dpi=300); plt.close()

    # ---- zbiorczy wykres 6-metryk ---------------------------------
    plot_big_bar(seq_df, old_df, gap_df)

    # ---- czasy + wykresy -------------------------------------
    plot_times(time_df, cvx_res, my_res, labels)

    # ---- macierze pomyłek ------------------------------------
    generate_confusions(labels, cvx_res, my_res)

    # ─────  ▼▼▼  DEBUG CALLS  ▼▼▼  ─────
    # Przykład użycia: wypisz pomyłki dwukropka ':' i cyfry '1'
    # list_mistakes_for_char("7", labels, my_res, field="date", system="My_model")
    # list_mistakes_for_char("7", labels, cvx_res, field="date", system="CV-X")
    # ─────────────────────────────────────

    # === Wypisanie średnich czasów i wykres słupkowy porównawczy ===
    avg = time_df.groupby("system")[["detect", "decode_date", "total"]].mean().round(4)

    print("\n=== Średnie czasy przetwarzania [s] ===")
    print(avg.rename(index={"My_model": "Model własny"}).rename_axis("System"))

    # Wykres słupkowy: średni czas działania
    labels = ["Czas przygotowania ROI", "Czas dekodowania (data)", "Czas całkowity"]
    x = np.arange(len(labels))
    width = 0.35

    my_vals = avg.loc["My_model", ["detect", "decode_date", "total"]]
    cvx_vals = avg.loc["CV-X", ["detect", "decode_date", "total"]]

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, my_vals, width, label="Model własny", color="tab:orange")
    plt.bar(x + width/2, cvx_vals, width, label="CV-X", color="tab:blue")

    plt.xticks(x, labels)
    plt.ylabel("Średni czas [s]")
    plt.title("Porównanie średnich czasów działania systemów wizyjnych")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT / "avg_time_bar_comparison.png", dpi=300)
    plt.close()

    print(f"\n✅  Wszystko zapisane w  {OUT}")

# ══════════════════════════════════════════════════════════════
if __name__=="__main__":
    main()
