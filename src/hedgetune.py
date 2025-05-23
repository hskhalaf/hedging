import numpy as np
import scipy.integrate as si
import numpy.random as rng
import matplotlib.pyplot as plt
import os
from scipy.optimize import brentq
import math
import seaborn as sns
os.chdir(os.path.dirname(__file__))

p = 12
C = (p/(p+1))**p * (1/(p+1))
g = lambda x: x**p * (1 - x) / C
r = lambda x: x

TAU_GRID      = np.linspace(0.0, 300.0, 800)
N_LIST        = np.array([2, 4, 8, 16, 32, 64, 128])
LAMBDA_GRID   = np.logspace(-4, 4, 600)
M_MC          = 120_000

def Z(lmb):
    return si.quad(lambda x: np.exp(lmb * g(x)), 0.0, 1.0, limit=200)[0]

def Eg_tilt(lmb, Z_lmb=None):
    if Z_lmb is None:
        Z_lmb = Z(lmb)
    num = si.quad(lambda x: g(x) * np.exp(lmb * g(x)), 0.0, 1.0, limit=200)[0]
    return num / Z_lmb

def KL_tilt(lmb, Eg=None, Z_lmb=None):
    if Z_lmb is None:
        Z_lmb = Z(lmb)
    if Eg is None:
        Eg = Eg_tilt(lmb, Z_lmb)
    return lmb * Eg - np.log(Z_lmb)

Z_vals    = np.array([Z(l) for l in TAU_GRID])
Eg_vals   = np.array([Eg_tilt(l, Zv) for l, Zv in zip(TAU_GRID, Z_vals)])
KL_vals   = np.array([KL_tilt(l, Eg, Zv) for l, Eg, Zv in zip(TAU_GRID, Eg_vals, Z_vals)])

def Eg_BoN(n):
    return n / (C * (n + p) * (n + p + 1))

def KL_BoN(n):
    return np.log(n) - 1.0 + 1.0 / n

Eg_n = np.array([Eg_BoN(n) for n in N_LIST])
KL_n = np.array([KL_BoN(n) for n in N_LIST])

def softbon_mc(n, tau_grid, m=M_MC):
    proxy_samples = rng.random((m, n))
    gold_samples  = g(proxy_samples)
    result_Eg = []
    result_KL = []
    bins      = np.linspace(0.0, 1.0, 501)
    bw        = bins[1] - bins[0]

    for tau in tau_grid:
        logits  = proxy_samples / tau
        gumbels = -np.log(-np.log(rng.random(logits.shape)))
        idx     = np.argmax(logits + gumbels, axis=1)
        chosen_g = gold_samples[np.arange(m), idx]
        result_Eg.append(np.mean(chosen_g))
        hist, _   = np.histogram(proxy_samples[np.arange(m), idx], bins=bins)
        q_pdf     = hist / (m * bw)
        q_pdf     = q_pdf[q_pdf > 0]
        result_KL.append(np.sum(q_pdf * np.log(q_pdf)) * bw)

    return np.array(result_KL), np.array(result_Eg)

softbon_curves = {n: softbon_mc(n, LAMBDA_GRID) for n in N_LIST}

def q_lambda_pdf(x, lmb):
    return (lmb * x + 1.0) * np.exp(lmb * (x - 1.0))

def Eg_BoP(lmb):
    return si.quad(lambda x: g(x) * q_lambda_pdf(x, lmb), 0.0, 1.0, limit=200)[0]

def KL_BoP(lmb, Eg=None):
    if Eg is None:
        Eg = Eg_BoP(lmb)
    integrand = lambda x: q_lambda_pdf(x, lmb) * (
        np.log(lmb * x + 1.0) + lmb * (x - 1.0)
    )
    return si.quad(integrand, 0.0, 1.0, limit=200)[0]

Eg_bop = np.array([Eg_BoP(l) for l in TAU_GRID])
KL_bop = np.array([KL_BoP(l, Eg) for l, Eg in zip(TAU_GRID, Eg_bop)])

def Z_proxy(lmb):
    return 1.0 if lmb == 0 else (np.exp(lmb) - 1.0) / lmb

def Eg_proxy_tilt(lmb, Zp=None):
    if Zp is None:
        Zp = Z_proxy(lmb)
    num = si.quad(lambda x: g(x) * np.exp(lmb * r(x)), 0.0, 1.0, limit=200)[0]
    return num / Zp

def KL_proxy_tilt(lmb, Zp=None):
    if Zp is None:
        Zp = Z_proxy(lmb)
    num_r = si.quad(lambda x: r(x) * np.exp(lmb * r(x)), 0.0, 1.0, limit=200)[0]
    E_r   = num_r / Zp
    return lmb * E_r - np.log(Zp)

Eg_proxy_vals = np.array([Eg_proxy_tilt(l) for l in TAU_GRID])
KL_proxy_vals = np.array([KL_proxy_tilt(l) for l in TAU_GRID])

from scipy.special import factorial

from scipy.special import gamma, gammainc

def hedge_tune(r_t, method, theta_min, theta_max, n=None, M=5000, tol=1e-6):
    u  = np.linspace(1e-8, 1-1e-8, M)
    du = u[1] - u[0]

    def R(theta):
        if method == 'BoN':
            s = 1/theta + np.log(u)
            p = u**(theta-1)

        elif method == 'SBoN':
            if theta == 0:
                Z   = 1.0/n
                E_u = n/(n+1)
            else:
                inc_n  = gamma(n)*gammainc(n, theta)
                inc_n1 = gamma(n+1)*gammainc(n+1, theta)
                Z      = inc_n/theta**n
                E_u    = inc_n1/(theta*inc_n)
            s = u - E_u
            p = u**(n-1)*np.exp(theta*u)/Z

        else:  # BoP
            q0 = (theta*u + 1)*np.exp(theta*(1-u))
            Zp = 1.0 if theta == 0 else (2*np.exp(theta)-(theta+2))/theta
            p  = q0/Zp
            s  = u + (u-1)/(theta*u+1)

        return np.sum(r_t(u)*s*p)*du

    if method == 'SBoN':
        Rvals = np.array([R(t) for t in LAMBDA_GRID])
        mask  = np.isfinite(Rvals)
        idx   = np.where((Rvals[:-1]*Rvals[1:]<=0)&mask[:-1]&mask[1:])[0]
        if not len(idx): return np.nan
        a, b = LAMBDA_GRID[idx[0]], LAMBDA_GRID[idx[0]+1]
    else:
        a, b = theta_min, theta_max
        fa, fb = R(a), R(b)
        for _ in range(100):
            if fa*fb <= 0: break
            a /= 2; b *= 2
            fa, fb = R(a), R(b)
        else:
            return np.nan

    root = brentq(R, a, b, xtol=tol)
    return round(root) if method == 'BoN' else root


theta_bon = hedge_tune(g, 'BoN', 2, 128)
klb, ebo  = KL_BoN(theta_bon), Eg_BoN(theta_bon)

theta_bop = hedge_tune(g, 'BoP', TAU_GRID[0], TAU_GRID[-1])
klp, ebp  = KL_BoP(theta_bon), Eg_BoP(theta_bon)

print(f"BoN: {theta_bon}, BoP: {theta_bop}")
sns.set_theme(style="whitegrid",palette="rocket");light_purple_gray="#f5f3f8"
fig, ax = plt.subplots(figsize=(12,6))
ax.set_facecolor(light_purple_gray)
ax.plot(KL_vals,      Eg_vals,        lw=2, alpha=0.8, label="Tilted (true)")
ax.plot(KL_proxy_vals,Eg_proxy_vals,  lw=2, alpha=0.8, label="Tilted (proxy)")
colors = plt.cm.viridis(np.linspace(0.15,0.85,len(N_LIST)))
for c,n in zip(colors, N_LIST):
    kl,eg = softbon_curves[n]
    ax.plot(kl, eg, lw=1.5, alpha=0.8, label=f"SBoN n={n}", color=c)
    i = np.where(N_LIST==n)[0][0]
    ax.scatter(KL_n[i], Eg_n[i], marker='o', color=c, s=50)
ax.plot(KL_bop, Eg_bop, lw=2, linestyle=":", alpha=0.8, label="BoP")
x0 = KL_proxy_vals[np.argmax(Eg_proxy_vals)]
ax.axvline(x0, linestyle="--")
ax.scatter(x0, Eg_proxy_vals.max(), marker="x", s=80)
ax.set_xlim(0, KL_proxy_vals[-1])
ax.set_xlabel("KL divergence to the reference distribution")
ax.set_ylabel("Expected value of true reward")

colors = ['gold', 'blue']
ax.scatter([klb], [ebo], marker='*', s=200, c=colors[0], edgecolors='black',label='BoN (HedgeTune)')
ax.scatter([klp], [ebp], marker='*', s=200, c=colors[1], edgecolors='black',label='BoP (HedgeTune)')

ax.legend()
plt.tight_layout()
plt.savefig("rewardKL.pdf", bbox_inches="tight")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LAMBDA_LIST = [0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 1]
recip_2sf = [float(f"{1/x:.2g}") for x in LAMBDA_LIST]
N_FINE = np.arange(2, 80)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(LAMBDA_LIST)))

def softbon_fixed_lambda(n_array, lmb, m=M_MC):
    kl_vals, eg_vals = [], []
    bins = np.linspace(0, 1, 501)
    bw = bins[1] - bins[0]
    for n in n_array:
        proxy = rng.random((m, n))
        gold = g(proxy)
        logits = proxy / lmb
        gumbels = -np.log(-np.log(rng.random(logits.shape)))
        idx = np.argmax(logits + gumbels, axis=1)
        chosen = gold[np.arange(m), idx]
        eg_vals.append(chosen.mean())
        hist, _ = np.histogram(proxy[np.arange(m), idx], bins=bins)
        pdf = hist / (m * bw)
        pdf = pdf[pdf > 0]
        kl_vals.append((pdf * np.log(pdf)).sum() * bw)
    return np.array(kl_vals), np.array(eg_vals)


sns.set_theme(style="whitegrid",palette="rocket");light_purple_gray="#f5f3f8"
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor(light_purple_gray)

Eg_n = np.array([Eg_BoN(n) for n in N_FINE])
KL_n = np.array([KL_BoN(n) for n in N_FINE])
ax.scatter(KL_n, Eg_n, c="black", s=10, label="BoN")

for c, lmb, temp in zip(colors, LAMBDA_LIST, recip_2sf):
    kl, eg = softbon_fixed_lambda(N_FINE, lmb)
    ax.plot(kl, eg, "-o", ms=3, lw=1.2, color=c, label=f"λ={temp}")


ax.set_xlabel("KL divergence to the reference distribution", fontsize=16)
ax.set_ylabel("Expected value of true reward",     fontsize=16)
ax.legend(loc="center right", bbox_to_anchor=(1.2, 0.263), ncol=1, fontsize=12)
plt.tight_layout()
plt.savefig("rewardKL2.pdf", bbox_inches="tight")
