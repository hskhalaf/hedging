import os, numpy as np, torch, seaborn as sns, matplotlib.pyplot as plt, matplotlib.lines as mlines
from tqdm.auto import tqdm
from scipy.optimize import brentq

dir_path = os.path.join(os.getcwd(), "converts")
SIZE = "80k"
files = sorted(f for f in os.listdir(dir_path) if f.endswith(".npz") and "25" in f and (SIZE in f))
K, M = 800, 1200
default_NS = np.arange(1, 40)
max_n = int(default_NS.max())
temps = np.concatenate(([0.0], np.logspace(-3, 3, 100)))
mu_vals = np.linspace(0, max_n - 1, len(temps))
F, L = len(files), len(temps)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
curves = np.zeros((L, F, len(default_NS)), dtype=np.float32)
avg_gold = np.zeros((L, F), dtype=np.float32)
best_n_star = np.zeros(F, dtype=np.float32)

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma, gammainc

def find_theta_star(proxy_scores, gold_scores, method='BoN', theta_min=1.0, theta_max=49.0, n=None, M=5000, tol=1e-6):
    K,S = proxy_scores.shape
    U_rows, G_rows = [], []
    for i in range(K):
        ranks = np.argsort(np.argsort(proxy_scores[i])); U = (ranks+1)/S
        order = np.argsort(proxy_scores[i]);     G = gold_scores[i][order]
        U_rows.append(U); G_rows.append(G)
    def R(theta):
        total = 0.0
        for U, G in zip(U_rows, G_rows):
            if method == 'BoN':
                s = 1/theta + np.log(U);       w = U**(theta-1)
            elif method == 'SBoN':
                if n is None: raise ValueError("n required for SBoN")
                if theta == 0:
                    Z, E_u = 1.0/n, n/(n+1)
                else:
                    inc_n  = gamma(n)   * gammainc(n,   theta)
                    inc_n1 = gamma(n+1) * gammainc(n+1, theta)
                    Z      = inc_n/theta**n; E_u = inc_n1/(theta*inc_n)
                s = U - E_u; w = U**(n-1)*np.exp(theta*U)/Z
            elif method == 'BoP':
                q0 = (theta*U+1)*np.exp(theta*(1-U))
                Zp = 1.0 if theta==0 else (2*np.exp(theta)-(theta+2))/theta
                w, s = q0/Zp, U + (U-1)/(theta*U+1)
            else:
                raise ValueError(f"Unknown method: {method}")
            p = w/np.sum(w); total += np.dot(G, s*p)
        return total/K
    if method == 'SBoN':
        grid = np.linspace(theta_min, theta_max, M)
        Rvals = np.array([R(t) for t in grid])
        mask = np.isfinite(Rvals)
        idxs = np.where(mask[:-1] & mask[1:] & (Rvals[:-1]*Rvals[1:]<=0))[0]
        if idxs.size==0: return np.nan
        a,b = grid[idxs[0]], grid[idxs[0]+1]
    else:
        a,b = theta_min, theta_max; fa,fb = R(a),R(b)
        for _ in range(100):
            if fa*fb<=0: break
            a/=2; b*=2; fa,fb = R(a),R(b)
        else: return np.nan
    root = brentq(R, a, b, xtol=tol)
    return round(root) if method=='BoN' else root

for fi, fname in enumerate(tqdm(files, desc="Files")):
    data = np.load(os.path.join(dir_path, fname), mmap_mode="r")
    proxy = torch.from_numpy(data["proxy"][:K, :max_n].astype(np.float32)).to(device)
    gold = torch.from_numpy(data["gold"][:K, :max_n].astype(np.float32)).to(device)
    ranks = proxy.argsort(dim=1).argsort(dim=1)
    proxy = ranks.float() / (max_n - 1)
    perms = torch.stack([torch.randperm(max_n, device=device) for _ in range(M)], dim=0)
    pvals = proxy.unsqueeze(0).expand(M, K, max_n).gather(2, perms.unsqueeze(1).expand(M, K, max_n))
    gvals = gold.unsqueeze(0).expand(M, K, max_n).gather(2, perms.unsqueeze(1).expand(M, K, max_n))
    for li, lam in enumerate(temps):
        for j, n in enumerate(default_NS):
            sp = pvals[:, :, :n]
            sg = gvals[:, :, :n]
            if lam == 0.0:
                best = sp.argmax(dim=2)
                eg = sg.gather(2, best.unsqueeze(2)).squeeze(2)
            else:
                lg = sp / lam
                m_ = lg.amax(dim=2, keepdim=True)
                ex = (lg - m_).exp()
                pr = ex / ex.sum(dim=2, keepdim=True)
                eg = (pr * sg).sum(dim=2)
            curves[li, fi, j] = eg.mean().item()
        P = 1
        Np = np.random.poisson(mu_vals[li], size=(M, P))
        trial_scores = np.zeros(M, dtype=np.float32)
        for m in range(M):
            scores = []
            for k in range(P):
                n_m = min(int(Np[m, k]) + 1, max_n)
                idx = pvals[m, :, :n_m].argmax(dim=1)
                scores.append(gvals[m, torch.arange(K, device=device), idx].mean().cpu().item())
            trial_scores[m] = np.mean(scores)
        avg_gold[li, fi] = trial_scores.mean()

li0 = int(np.argmin(np.abs(temps - 0.0)))
colors = sns.color_palette("rocket", n_colors=F)
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor("#FBFAFD")
for fi in range(F):
    ax.scatter(default_NS, curves[li0, fi], marker='o', s=40, alpha=0.5, color=colors[fi])
    ax.plot(default_NS, curves[:, fi, :].max(axis=0), '--', linewidth=2, alpha=0.7, color=colors[fi])
    ax.plot(mu_vals + 1, avg_gold[:, fi], '-', linewidth=2, alpha=1, color=colors[fi])
    n_star_int = int(best_n_star[fi]) if not np.isnan(best_n_star[fi]) and 1 <= best_n_star[fi] <= max_n else default_NS[curves[li0, fi].argmax()]
    bon_y = curves[li0, fi, n_star_int - 1]

ax.set_xlabel("Number of samples $n$", fontsize=15)
ax.set_ylabel("Expected true reward", fontsize=15)
ax.grid(which="both", linestyle="--", alpha=0.5)

handles=[
    mlines.Line2D([],[],marker='o',linestyle='None',color='black',alpha=0.3,markersize=6,label='BoN'),
    mlines.Line2D([],[],linestyle='-',linewidth=2,color='black',alpha=0.7,label='BoP'),
    mlines.Line2D([],[],linestyle='--',linewidth=2,color='black',alpha=0.7,label='SBoN'),
]
import re

file_handles = []
for i, fname in enumerate(files):
    m = re.search(r"(\d+k)_seed(\d+)", fname)
    if m:
        label = f"{m.group(1)} (Seed {m.group(2)})"
    else:
        label = fname
    file_handles.append(
        mlines.Line2D(
            [], [], marker='s', linestyle='None',
            color=colors[i], markersize=8,
            label=label
        )
    )

fig.subplots_adjust(right=0.75)
all_handles = handles + file_handles
leg = ax.legend(
    handles=all_handles,
    loc='upper left',
    bbox_to_anchor=(1.02, 0.3),
    frameon=True
)
fig.savefig(
    os.path.join(dir_path, f"{SIZE}.pdf"),
    bbox_inches='tight',
    pad_inches=0.1,
    bbox_extra_artists=[leg]
)

