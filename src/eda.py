"""Task 2 - three exploratory plots from the RAW (not standardised) data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_raw
from prepare import clean, ROOT
from tables import RESULTS

RESULTS.mkdir(exist_ok=True)
df, _ = clean(load_raw(str(ROOT / "data")))

# 2.1  currents vs row number (first 2000 rows), fault shaded
sub = df.iloc[:2000]
fig, ax = plt.subplots(figsize=(12, 4))
for c in ["Ia", "Ib", "Ic"]:
    ax.plot(sub.index, sub[c], label=c, linewidth=0.8)
ax.fill_between(sub.index, 0, 1, where=(sub["fault"].to_numpy() == 1),
                transform=ax.get_xaxis_transform(), color="red", alpha=0.2,
                step="mid", label="fault = 1")
ax.set_xlabel("row number (time sample)"); ax.set_ylabel("current (A)")
ax.set_title("Line currents, first 2000 rows (red = fault)")
ax.legend(loc="upper right"); fig.tight_layout()
fig.savefig(RESULTS / "fig1a_currents_vs_row.png", dpi=150); plt.close(fig)

# 2.2  scatter Ia vs Ib
fig, ax = plt.subplots(figsize=(6, 5))
for lab, col, name in [(0, "tab:blue", "no fault"), (1, "tab:red", "fault")]:
    m = df["fault"] == lab
    ax.scatter(df.loc[m, "Ia"], df.loc[m, "Ib"], s=4, alpha=0.4, c=col, label=name)
ax.set_xlabel("Ia (A)"); ax.set_ylabel("Ib (A)"); ax.set_title("Ia vs Ib")
ax.legend(markerscale=4); fig.tight_layout()
fig.savefig(RESULTS / "fig1b_scatter_Ia_Ib.png", dpi=150); plt.close(fig)

# 2.3  histogram of Ia
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(df["Ia"].min(), df["Ia"].max(), 60)
ax.hist(df.loc[df.fault == 0, "Ia"], bins=bins, alpha=0.6, color="tab:blue", label="no fault")
ax.hist(df.loc[df.fault == 1, "Ia"], bins=bins, alpha=0.6, color="tab:red", label="fault")
ax.set_xlabel("Ia (A)"); ax.set_ylabel("number of rows"); ax.set_title("Histogram of Ia")
ax.legend(); fig.tight_layout()
fig.savefig(RESULTS / "fig1c_hist_Ia.png", dpi=150); plt.close(fig)
print("Saved fig1a, fig1b, fig1c in results/")
