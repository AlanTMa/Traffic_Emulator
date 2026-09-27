"""
Numbers and figures behind docs/azure_trace_assessment.md.

    python -m scripts.azure_trace TRACE.txt OUT_DIR

TRACE.txt is AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt from
github.com/Azure/AzurePublicDataset (CC BY 4.0; not redistributed here).
Writes summary.txt, concentration.png, durations.png and rates.png. Reads
the trace only; nothing here feeds the emulator.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_OFFERED = 60.43       # MB/s offered by the paper's 5x3 instance
WINDOW = 5.0                # the emulator's default round, s

def cumulative_share(counts):
    counts = np.sort(np.asarray(counts, dtype=float))[::-1]
    return np.cumsum(counts) / counts.sum()

def needed(share, q):
    return int((share < q).sum()) + 1

def main(path, out):
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    df = pd.read_csv(path, dtype={"app": "category", "func": "category",
                                  "end_timestamp": "float64", "duration": "float64"})
    df["start"] = df["end_timestamp"] - df["duration"]
    span = float(np.ceil(df["end_timestamp"].max()))
    n_hours = int(np.ceil(span / 3600))
    rate = len(df) / span
    counts = df["app"].value_counts()
    say(f"rows {len(df):,}; apps {len(counts)}; (app, function) pairs "
        f"{df.groupby(['app', 'func'], observed=True).ngroups}; span {span / 86400:.2f} days; "
        f"{rate:.2f} invocations/s")

    say()
    say("== skew")
    share = cumulative_share(counts)
    say("share of invocations, top 1 / 2 / 5 / 10 / 20 / 50 apps: "
        + " / ".join(f"{share[k - 1]:.3f}" for k in (1, 2, 5, 10, 20, 50)))
    say("apps carrying 50 / 90 / 99 / 99.5%: "
        + " / ".join(str(needed(share, q)) for q in (0.5, 0.9, 0.99, 0.995)))
    busy = df.groupby("app", observed=True)["duration"].sum()
    wshare = cumulative_share(busy)
    say("share of busy time (sum of durations), top 1 / 2 / 5 / 10 apps: "
        + " / ".join(f"{wshare[k - 1]:.3f}" for k in (1, 2, 5, 10))
        + f"; apps carrying 99.5%: {needed(wshare, 0.995)}")
    busy_rank = busy.rank(ascending=False).astype(int)
    say("busy-time rank of the top 5 apps by invocations: "
        + ", ".join(str(busy_rank[a]) for a in counts.index[:5]))
    pairs = df.groupby(["app", "func"], observed=True).size()
    pshare = cumulative_share(pairs)
    say("share of invocations, top 5 / 10 / 20 pairs: "
        + " / ".join(f"{pshare[k - 1]:.3f}" for k in (5, 10, 20))
        + f"; pairs carrying 99.5%: {needed(pshare, 0.995)}")
    c = np.sort(counts.to_numpy())
    gini = 1 - 2 * np.sum(np.cumsum(c) / c.sum()) / len(c) + 1 / len(c)
    say(f"Gini over apps {gini:.3f}; median app {counts.median():.0f} invocations in the trace; "
        f"{int((counts < 100).sum())} of {len(counts)} apps below 100")
    rates = counts / span
    say(f"apps averaging at least 1/s: {int((rates >= 1).sum())}; 0.1/s: {int((rates >= 0.1).sum())}; "
        f"1/min: {int((rates >= 1 / 60).sum())}")
    df["hour"] = (df["start"] // 3600).astype(int)
    active = df.groupby("app", observed=True)["hour"].nunique() / n_hours
    say(f"apps with an invocation in at least 90% of hours: {int((active >= 0.9).sum())}; "
        f"50%: {int((active >= 0.5).sum())}; under 10%: {int((active < 0.1).sum())}")
    funcs = df.groupby("app", observed=True)["func"].nunique()
    say(f"functions per app: median {funcs.median():.0f}, max {funcs.max()}")

    say()
    say("== top 10 apps, arrivals over time (1-minute bins; Poisson: CV 1/sqrt(mean), IoD 1, CV iat 1)")
    say(f"{'app':>8} {'rate/s':>7} {'active':>6} {'CV':>6} {'peak/mean':>9} {'zero':>5} {'IoD':>7} "
        f"{'CV iat':>7} {'burst':>5} {'gap':>5} {'at gap':>6} {'funcs':>5}")
    minute_bins = np.arange(0, span + 60, 60)
    profiles = {}
    for app in counts.index[:10]:
        s = np.sort(df.loc[df["app"] == app, "start"].to_numpy())
        per_min = np.histogram(s, bins=minute_bins)[0]
        profiles[app] = per_min
        iat = np.diff(s)
        burst = np.mean(iat < 0.05)
        gaps = np.round(iat[iat >= 1.0])
        values, freq = np.unique(gaps, return_counts=True)
        gap = values[np.argmax(freq)]
        at_gap = np.mean(np.abs(iat - gap) <= 0.02 * gap)
        say(f"{str(app)[:8]:>8} {len(s) / span:>7.2f} {active[app]:>6.2f} {per_min.std() / per_min.mean():>6.2f} "
            f"{per_min.max() / per_min.mean():>9.1f} {np.mean(per_min == 0):>5.2f} "
            f"{per_min.var() / per_min.mean():>7.1f} {iat.std() / iat.mean():>7.1f} {burst:>5.2f} "
            f"{gap:>5.0f} {at_gap:>6.2f} {funcs[app]:>5}")
    say("active: share of hours with an invocation; burst: share of inter-arrivals under 50 ms; "
        "gap: most common inter-arrival of at least 1 s, and the share within 2% of it")
    iat = df.sort_values(["app", "start"]).groupby("app", observed=True)["start"].diff().dropna()
    say(f"same-app inter-arrivals: {np.mean(iat < 0.01):.2f} under 10 ms, {np.mean(iat < 1):.2f} under 1 s, "
        f"{np.mean(iat > 60):.2f} over 1 min; median {iat.median():.2f} s")

    say()
    say("== cycles")
    by_hour = df.groupby((df["start"] % 86400) // 3600).size() / (span / 86400) / 3600
    say(f"aggregate rate by hour of day: min {by_hour.min():.2f}/s, max {by_hour.max():.2f}/s, "
        f"max/min {by_hour.max() / by_hour.min():.2f}")
    by_day = df.groupby(df["start"] // 86400).size()
    say("invocations per day: " + ", ".join(f"{int(v):,}" for v in by_day) + f"; max/min {by_day.max() / by_day.min():.1f}")
    top_days = df[df["app"].isin(counts.index[:10])].groupby(["app", df["start"] // 86400], observed=True).size().unstack(fill_value=0)
    say("top 10 apps, days without an invocation: " + ", ".join(str(int(v)) for v in (top_days == 0).sum(axis=1)[counts.index[:10]]))

    say()
    say(f"== the top app in {WINDOW:.0f} s windows")
    s = np.sort(df.loc[df["app"] == counts.index[0], "start"].to_numpy())
    per_window = np.histogram(s, bins=np.arange(0, span + WINDOW, WINDOW))[0]
    say(f"real time: mean {per_window.mean():.2f} per window, empty windows {np.mean(per_window == 0):.2f}, max {per_window.max()}")
    squeeze = PAPER_OFFERED / rate
    per_window = np.histogram(s / squeeze, bins=np.arange(0, span / squeeze + WINDOW, WINDOW))[0]
    say(f"time compressed {squeeze:.0f}x (one unit per invocation, the paper's {PAPER_OFFERED} units/s in aggregate): "
        f"mean {per_window.mean():.1f}, CV {per_window.std() / per_window.mean():.2f} "
        f"(Poisson {1 / np.sqrt(per_window.mean()):.2f}), empty windows {np.mean(per_window == 0):.2f}, max {per_window.max()}")

    say()
    say("== durations (s)")
    d_all = df["duration"].to_numpy()
    d = d_all[d_all > 0]
    q = np.percentile(d, [50, 90, 95, 99, 99.9])
    say(f"recorded as 0: {np.mean(d_all <= 0):.3f} of rows; of the rest, exactly 1 ms: {np.mean(d <= 0.001):.3f} (1 ms resolution)")
    say(f"mean {d.mean():.3f}, median {q[0]:.3f}, p90 {q[1]:.2f}, p95 {q[2]:.1f}, p99 {q[3]:.1f}, p99.9 {q[4]:.0f}, max {d.max():.0f}")
    say(f"CV {d.std() / d.mean():.2f} (exponential 1.00); under 10 ms {np.mean(d < 0.01):.3f}, under 100 ms {np.mean(d < 0.1):.3f}, "
        f"over 1 s {np.mean(d > 1):.3f}, over 60 s {np.mean(d > 60):.4f}")
    means = df[df["duration"] > 0].groupby("app", observed=True)["duration"].mean()
    say(f"per-app mean: min {means.min():.4f}, median {means.median():.2f}, max {means.max():.0f}")
    for app in counts.index[:5]:
        da = df.loc[df["app"] == app, "duration"].to_numpy()
        da = da[da > 0]
        say(f"  {str(app)[:8]}: mean {da.mean():.3f}, median {np.median(da):.3f}, CV {da.std() / da.mean():.1f}")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for values, label in ((share, "apps, by invocations"), (wshare, "apps, by busy time"),
                          (pshare, "(app, function) pairs, by invocations")):
        ax.plot(np.arange(1, len(values) + 1), values, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("rank")
    ax.set_ylabel("cumulative share")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "concentration.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    edges = np.logspace(-3, 3, 31)          # 5 bins per decade
    edges[0] = 0.9e-3                       # so the first bin holds the 1 ms floor
    expo = np.exp(-edges[:-1] / d.mean()) - np.exp(-edges[1:] / d.mean())
    ax[0].stairs(np.histogram(d, bins=edges)[0] / len(d), edges, fill=True, alpha=0.7,
                 label=f"trace, nonzero durations ({np.mean(d_all <= 0):.1%} are 0)")
    ax[0].stairs(expo, edges, color="k", linestyle="--", label=f"exponential, same mean ({d.mean():.2f} s)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("duration (s)")
    ax[0].set_ylabel("share of invocations per bin")
    ax[0].set_title("histogram, 5 bins per decade")
    ax[0].legend(fontsize=8)
    x = np.logspace(-3, 3, 300)
    d_sorted = np.sort(d)
    ax[1].plot(x, 1 - np.searchsorted(d_sorted, x, side="right") / len(d_sorted), label="trace")
    ax[1].plot(x, np.exp(-x / d.mean()), "k--", label="exponential, same mean")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-7, 1.5)
    ax[1].set_xlabel("duration x (s)")
    ax[1].set_ylabel("share of invocations longer than x")
    ax[1].set_title("tail")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "durations.png", dpi=130)

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    hourly = np.histogram(df["start"].to_numpy(), bins=np.arange(0, span + 3600, 3600))[0] / 3600
    ax[0].plot(np.arange(len(hourly)) / 24, hourly, lw=0.8)
    ax[0].set_ylabel("all apps, invocations/s per hour")
    for app in counts.index[:5]:
        ax[1].plot(minute_bins[:-1] / 86400, profiles[app], lw=0.5, label=str(app)[:8])
    ax[1].set_ylabel("top 5 apps, invocations per minute")
    ax[1].set_xlabel("day")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "rates.png", dpi=130)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
