"""Synthetic vector-PDF pump sheet with known curves (R2500 head from the user's catalogue)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = [55.7137788582101, -0.0129268151217755, 2.59497540663193e-05, -1.81390921025721e-08,
     4.68161524135285e-12, -4.37675172151609e-16]
P = [0.62, 1.9e-4, 1.1e-7, -3.0e-11]           # hp/stage, made up but plausible
head = lambda q: np.polynomial.polynomial.polyval(q, H)
power = lambda q: np.polynomial.polynomial.polyval(q, P)
eff = lambda q: 100 * q * head(q) / (135770 * power(q))
q = np.linspace(0, 3500, 400)
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(q, head(q), color="#1f3cc8", lw=2); ax.set_ylim(0, 70); ax.set_xlim(0, 4000)
ax.set_xlabel("Capacity, bbl/d"); ax.set_ylabel("Head (ft)")
ax.grid(True, color="0.75", lw=0.6); ax.set_xticks(range(0, 4001, 500)); ax.set_yticks(range(0, 71, 10))
a2 = ax.twinx(); a2.plot(q, power(q), color="#d42020", lw=2); a2.set_ylim(0, 1.4); a2.set_ylabel("Power (hp)")
a2.set_yticks(np.arange(0, 1.41, 0.2))
a3 = ax.twinx(); a3.spines["right"].set_position(("axes", 1.12)); a3.plot(q, eff(q), color="#1a8a2a", lw=2)
a3.set_ylim(0, 70); a3.set_ylabel("Efficiency (%)"); a3.set_yticks(range(0, 71, 10))
ax.set_title("TEST PUMP T2500 60 Hz 3500 rpm (synthetic)")
fig.subplots_adjust(right=0.8)
fig.savefig(__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "T2500_vector.pdf"); print("ok")
