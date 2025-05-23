import os, json, numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns, pandas as pd, time
from matplotlib.widgets import Slider
from tqdm.auto import tqdm
from scipy.optimize import brentq
from scipy.stats import rankdata, kstest

torch.set_default_dtype(torch.float32)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FILE = "data/candidates_results.json"
CACHE = FILE + ".metrics.npz"
LAMBDAS = np.linspace(0, 20, 300, dtype=np.float32)
MU_VALUES = np.linspace(0, 50, 250, dtype=np.float32)
BOOTSTR = 500
zcrit = 1.96
PEAK_MIN, PEAK_MAX, PEAK_STEP = 80, 99, 5
PEAK_VALUES = np.arange(PEAK_MIN, PEAK_MAX + 1, PEAK_STEP)

def phi(n):
    if n <= 1:
        return -np.inf
    safe_n = np.maximum(n, 1e-9)
    return np.log(safe_n) - (safe_n - 1) / safe_n

Nmax, K = 10000, 500
phi_1, phi_nmax = 0, phi(Nmax)
if phi_nmax <= phi_1:
    phi_nmax = phi_1 + 1.0
x_targets = np.linspace(phi_1, phi_nmax, K)

def invert_phi(x):
    if x <= phi_1:
        return 1
    try:
        return int(np.round(brentq(lambda n: phi(n) - x, 1.0001, max(Nmax, 1.0002))))
    except ValueError:
        return Nmax if x > phi(Nmax) + 1e-9 else 1

N_SIZES = np.unique([invert_phi(xt) for xt in x_targets])
N_SIZES = N_SIZES[N_SIZES >= 1]
if len(N_SIZES) < 2:
    N_SIZES = np.array([1, 100])
if N_SIZES[0] != 1:
    N_SIZES = np.insert(N_SIZES, 0, 1)
Nmax_actual = N_SIZES.max()
valid_plot_indices_n = N_SIZES > 0
N_SIZES_valid = N_SIZES[valid_plot_indices_n]
if len(N_SIZES_valid) == 0:
    raise ValueError("No valid N_SIZES > 0 found.")

with open(FILE, "r") as f:
    obj = json.load(f)
scores_np_orig = {"train": np.asarray(obj[0]["scores_2b"], dtype=np.float32)}
train_np_orig = scores_np_orig["train"]
n_total = len(train_np_orig)
if n_total == 0:
    raise ValueError("Loaded train scores are empty!")

rng = np.random.default_rng()
full_paths = torch.tensor(rng.integers(n_total, size=(BOOTSTR, Nmax_actual)), device=DEVICE)
uniform = torch.rand(BOOTSTR, device=DEVICE)
lam_t = torch.tensor(LAMBDAS, device=DEVICE)
L, J = len(LAMBDAS), len(N_SIZES)
results_cache = {}

def calculate_all_probs_for_peak(pctl_peak_value):
    train_np = scores_np_orig["train"]
    n_total = len(train_np)
    sort_indices = np.argsort(train_np)
    unsort_indices = np.argsort(sort_indices)
    train_np_sorted = train_np[sort_indices]
    idx_thresh = min(n_total - 1, int(n_total * pctl_peak_value / 100))
    val_shape_sorted = np.zeros_like(train_np_sorted, dtype=np.float32)
    if idx_thresh > 0:
        val_shape_sorted[:idx_thresh] = np.linspace(0, 1, idx_thresh, endpoint=False, dtype=np.float32)
    if idx_thresh < n_total:
        val_shape_sorted[idx_thresh:] = np.linspace(1, 0, n_total - idx_thresh, endpoint=True, dtype=np.float32)
    val_shape_mean, val_shape_std = val_shape_sorted.mean(), val_shape_sorted.std()
    train_np_mean, train_np_std = train_np.mean(), train_np.std()
    val_rescaled_sorted = ((val_shape_sorted - val_shape_mean) / val_shape_std) * train_np_std + train_np_mean
    val_synthetic_np = val_rescaled_sorted[unsort_indices].astype(np.float32)
    current_scores_np = {"train": scores_np_orig["train"], "val": val_synthetic_np}
    sorted_all = {k: np.sort(v) for k, v in current_scores_np.items()}
    q = lambda x, s: np.asarray(np.searchsorted(s, x, "right") / len(s), dtype=np.float32)
    C_current = {k: torch.tensor(q(v, sorted_all[k]), device=DEVICE) for k, v in current_scores_np.items()}
    train_full_current = C_current["train"]
    val_full_current = C_current["val"]
    def Eg_BoN_sim(n):
        idx = full_paths[:, :n].to(DEVICE)
        proxy_n = train_full_current[idx]
        gold_n = val_full_current[idx]
        picks = proxy_n.argmax(dim=1)
        return gold_n[torch.arange(BOOTSTR, device=DEVICE), picks].mean().item()
    def tune_n_for_gold(g, n_min, n_max):
        lo, hi = n_min, n_max
        if Eg_BoN_sim(lo) >= g:
            return lo
        if Eg_BoN_sim(hi) < g:
            return hi
        while lo < hi:
            mid = (lo + hi) // 2
            if Eg_BoN_sim(mid) >= g:
                hi = mid
            else:
                lo = mid + 1
        return lo
    gold_matrix = val_full_current[full_paths]
    n_star = [[tune_n_for_gold(float(g), 2, Nmax_actual) for g in row] for row in gold_matrix]
    _prob_train = np.zeros((L + 1, J), np.float32)
    _prob_val = np.zeros((L + 1, J), np.float32)
    dev_calc = DEVICE
    indices_1 = full_paths[:, 0].to(dev_calc)
    t_mat_1 = train_full_current.to(dev_calc)[indices_1]
    v_mat_1 = val_full_current.to(dev_calc)[indices_1]
    sel_t_1 = t_mat_1.unsqueeze(0).expand(L, BOOTSTR)
    sel_v_1 = v_mat_1.unsqueeze(0).expand(L, BOOTSTR)
    _prob_train[:, 0] = 0.5
    _prob_val[:, 0] = 0.5
    for n_idx_loop, n in enumerate(N_SIZES[1:]):
        actual_storage_idx = n_idx_loop + 1
        dev_loop = torch.device("cpu") if n > 30000 else DEVICE
        indices_n = full_paths[:, :n].to(dev_loop)
        t_mat_n = train_full_current.to(dev_loop)[indices_n]
        v_mat_n = val_full_current.to(dev_loop)[indices_n]
        t_batched = t_mat_n.unsqueeze(0).expand(L, BOOTSTR, n)
        v_batched = v_mat_n.unsqueeze(0).expand(L, BOOTSTR, n)
        if n == 1:
            picks_base = torch.zeros((L, BOOTSTR), dtype=torch.long, device=dev_loop)
        else:
            lam_batched = lam_t.to(dev_loop).view(L, 1, 1)
            w = torch.softmax(lam_batched * t_batched, dim=2)
            cums = torch.cumsum(w, dim=2)
            uniform2 = uniform.to(dev_loop).unsqueeze(0)
            picks_base = (uniform2.unsqueeze(2) < (cums + 1e-9)).int().argmax(dim=2)
        picks_inf = t_mat_n.argmax(dim=1, keepdim=True).transpose(0, 1)
        picks_all = torch.cat([picks_base, picks_inf], dim=0)
        t_all = torch.cat([t_batched, t_mat_n.unsqueeze(0)], dim=0)
        v_all = torch.cat([v_batched, v_mat_n.unsqueeze(0)], dim=0)
        rows = torch.arange(BOOTSTR, device=dev_loop).unsqueeze(0).expand(L + 1, -1)
        sel_t = t_all.gather(2, picks_all.unsqueeze(2)).squeeze(2)
        sel_v = v_all.gather(2, picks_all.unsqueeze(2)).squeeze(2)
        base_t = sel_t_1.to(dev_loop)[0]
        base_v = sel_v_1.to(dev_loop)[0]
        gt_t = sel_t > 1.1 * base_t
        gt_v = sel_v > 1.1 * base_v
        _prob_train[: L + 1, actual_storage_idx] = gt_t.float().mean(dim=1).cpu().numpy()
        _prob_val[: L + 1, actual_storage_idx] = gt_v.float().mean(dim=1).cpu().numpy()
    P = len(MU_VALUES)
    _prob_train_pois = np.zeros(P, np.float32)
    _prob_val_pois = np.zeros(P, np.float32)
    baseline_t_rep = sel_t_1[0]
    baseline_v_rep = sel_v_1[0]
    for m_idx, mu in enumerate(MU_VALUES):
        raw_N_np = rng.poisson(mu, size=BOOTSTR)
        raw_N = torch.from_numpy(raw_N_np).to(dev_calc)
        Np1 = (raw_N + 1).clamp(min=1, max=N_SIZES.max()).long()
        proxy_max = torch.empty(BOOTSTR, device=dev_calc)
        gold_max = torch.empty(BOOTSTR, device=dev_calc)
        for b in range(BOOTSTR):
            n_i = Np1[b].item()
            idx = full_paths[b, :n_i].to(dev_calc)
            proxy_win = train_full_current[idx]
            gold_win = val_full_current[idx]
            max_val, argmax = proxy_win.max(dim=0)
            proxy_max[b] = max_val
            gold_max[b] = gold_win[argmax]
        gt_t_p = proxy_max > 1.1 * baseline_t_rep
        gt_v_p = gold_max > 1.1 * baseline_v_rep
        _prob_train_pois[m_idx] = gt_t_p.float().mean().cpu().item()
        _prob_val_pois[m_idx] = gt_v_p.float().mean().cpu().item()
    _se_train = np.sqrt(_prob_train * (1 - _prob_train) / BOOTSTR)
    _se_val = np.sqrt(_prob_val * (1 - _prob_val) / BOOTSTR)
    _ci_hw_t = zcrit * np.nan_to_num(_se_train)
    _ci_hw_v = zcrit * np.nan_to_num(_se_val)
    return _prob_train, _prob_val, _ci_hw_t, _ci_hw_v, _prob_train_pois, _prob_val_pois, n_star

for peak_val in tqdm(PEAK_VALUES, desc="Peak Pctl Pre-calculation"):
    results_cache[peak_val] = calculate_all_probs_for_peak(peak_val)

sns.set_theme(style="white", context="talk")
plt.rc("axes", edgecolor="black", linewidth=1)
sns.despine(trim=True)
P, N = len(PEAK_VALUES), len(N_SIZES_valid)
deltaP_pct_all = np.zeros((P, N), dtype=np.float32)
ci_delta_pct_all = np.zeros((P, N), dtype=np.float32)
for i, peak in enumerate(PEAK_VALUES):
    prob_train, prob_val, ci_hw_t, ci_hw_v, train_pois, val_pois, _ = results_cache[peak]
    soft_val = prob_val[:-1, valid_plot_indices_n]
    ci_soft = ci_hw_v[:-1, valid_plot_indices_n]
    bon_val = prob_val[-1, valid_plot_indices_n]
    best_idxs = soft_val.argmax(axis=0)
    P_at_best = soft_val[best_idxs, np.arange(N)]
    deltaP_pct_all[i] = (P_at_best - bon_val) * 100
    ci_delta_pct_all[i] = ci_soft[best_idxs, np.arange(N)] * 100
n_arr = N_SIZES_valid.astype(float)
x_axis = np.log(n_arr) - (n_arr - 1) / n_arr
SMOOTH = 15
sns.set_style("whitegrid")
sns.set_context("talk", rc={"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 10})
colors = sns.color_palette("rocket", n_colors=len(PEAK_VALUES))
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
ax.set_facecolor("#FBFAFD")
for y, ci, peak, color in zip(deltaP_pct_all, ci_delta_pct_all, PEAK_VALUES, colors):
    y_s = pd.Series(y).rolling(SMOOTH, center=True, min_periods=1).mean().values
    ci_s = pd.Series(ci).rolling(SMOOTH, center=True, min_periods=1).mean().values
    ax.plot(x_axis, y_s, color=color, linewidth=1.5, solid_capstyle="round", label=rf"$\alpha={peak}\%$")
    ax.fill_between(x_axis, y_s - ci_s, y_s + ci_s, color=color, alpha=0.15)
ax.set_xlabel("log(n) - (n-1)/n")
ax.set_ylabel("Reward Improvement over BoN (%)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.tick_params(labelsize=9)
ax.legend(loc="best")
plt.tight_layout()
fig.savefig("synth.pdf", bbox_inches="tight")
fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=150)
ax2.set_facecolor("#FBFAFD")
threshold_n = max(MU_VALUES) - 1
first = True
for c, (_, (prob_train, prob_val, _, _, train_pois, val_pois, _)) in zip(colors, results_cache.items()):
    label_tp = "Proxy Reward (BoP)" if first else "_nolegend_"
    label_vp = "True Reward (BoP)" if first else "_nolegend_"
    label_tb = "Proxy Reward (BoN)" if first else "_nolegend_"
    label_vb = "True Reward (BoN)" if first else "_nolegend_"
    ax2.plot(MU_VALUES, train_pois, linestyle=(0, (1, 1)), color=c, label=label_tp, alpha=0.5)
    ax2.plot(MU_VALUES, val_pois, linestyle='-', color=c, label=label_vp, alpha=0.5)
    imax = val_pois.argmax()
    ax2.scatter(MU_VALUES[imax], val_pois[imax], marker='*', s=120, color='blue', edgecolor='black', alpha=0.6, zorder=10, label = "BoP (Opt)")

    fixed_train = prob_train[-1, valid_plot_indices_n]
    fixed_val = prob_val[-1, valid_plot_indices_n]
    n_vals = N_SIZES_valid.astype(float)
    mask = n_vals <= threshold_n
    ax2.scatter(n_vals[mask], fixed_train[mask], marker="x", s=50, color=c, label=label_tb)
    ax2.scatter(n_vals[mask], fixed_val[mask], marker="o", s=50, color=c, label=label_vb)
    imax = fixed_val[mask].argmax()
    ax2.scatter(n_vals[mask][imax], fixed_val[mask][imax], marker='*', s=120, color='yellow', edgecolor='black', zorder=10, alpha=0.6,  label = "BoN (Opt)")
    first = False
ax2.set_xlabel("Average number of samples n")
ax2.set_ylabel("Reward Improvement over Base Distribution (%)")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.tick_params(labelsize=9)
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.0, 0.9))
plt.tight_layout()
fig2.savefig("synth2.pdf", bbox_inches="tight")
