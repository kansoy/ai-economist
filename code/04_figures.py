from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
GPT = "gpt-6.1-sol"
OPUS = "claude-opus-5-5"


def has_lualatex():
    return shutil.which("lualatex") is not None


@dataclass(frozen=True)
class MonochromeTheme:
    WIDTH_IN: float = 469.755 * 0.94 / 72.27
    PT: float = 72.0 / 72.27
    BLACK: str = "#111111"
    DARK: str = "#4A4A4A"
    MID: str = "#828282"
    LIGHT: str = "#D6D6D6"
    PALE: str = "#F3F3F3"
    GRID: str = "#E6E6E6"
    use_pgf: bool = False

    @contextmanager
    def context(self):
        p = self.PT
        rc = {
            "font.family": "serif",
            "font.serif": ["Source Serif Pro", "Source Serif 4", "DejaVu Serif"],
            "font.size": 8.6 * p,
            "axes.labelsize": 8.6 * p,
            "axes.titlesize": 8.9 * p,
            "xtick.labelsize": 8.0 * p,
            "ytick.labelsize": 8.1 * p,
            "legend.fontsize": 8.0 * p,
            "text.color": self.BLACK,
            "axes.labelcolor": self.BLACK,
            "xtick.color": self.BLACK,
            "ytick.color": self.BLACK,
            "axes.edgecolor": self.BLACK,
            "axes.linewidth": 0.50 * p,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.axisbelow": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5 * p,
            "xtick.major.width": 0.50 * p,
            "ytick.major.size": 0.0,
            "grid.color": self.GRID,
            "grid.linewidth": 0.40 * p,
            "lines.linewidth": 1.25 * p,
            "lines.markersize": 4.5 * p,
            "lines.markeredgewidth": 0.85 * p,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
        if self.use_pgf:
            rc.update(
                {
                    "pgf.texsystem": "lualatex",
                    "pgf.rcfonts": False,
                    "pgf.preamble": "\n".join(
                        [
                            r"\usepackage{fontspec}",
                            r"\setmainfont{SourceSerifPro}[Extension=.otf,"
                            r"UprightFont=*-Regular,BoldFont=*-Bold,ItalicFont=*-RegularIt,"
                            r"BoldItalicFont=*-BoldIt,Numbers={Lining,Tabular}]",
                        ]
                    ),
                }
            )
        with mpl.rc_context(rc):
            yield self

    def bold(self, text):
        if self.use_pgf:
            return rf"\textbf{{{text}}}"
        return text

    def pct(self, text):
        return text.replace("%", r"\%") if self.use_pgf else text

    @staticmethod
    def halo(width=2.2):
        return [pe.Stroke(linewidth=width, foreground="white"), pe.Normal()]

    def export(self, fig, stem):
        stem.parent.mkdir(parents=True, exist_ok=True)
        pdf_path = stem.with_suffix(".pdf")
        png_path = stem.with_suffix(".png")
        save_kw = dict(bbox_inches="tight", pad_inches=4.0 / 72.0, facecolor="white")
        if self.use_pgf:
            fig.savefig(pdf_path, backend="pgf", format="pdf", **save_kw)
            if shutil.which("pdftoppm"):
                subprocess.run(
                    ["pdftoppm", "-gray", "-png", "-r", "300", "-singlefile", str(pdf_path), str(stem)],
                    check=True,
                )
        else:
            fig.savefig(pdf_path, format="pdf", **save_kw)
            fig.savefig(png_path, format="png", dpi=300, **save_kw)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def lookup(rows, **match):
    hits = [row for row in rows if all(row.get(key) == val for key, val in match.items())]
    if len(hits) != 1:
        raise ValueError(f"Expected 1 match for {match}, found {len(hits)}")
    return hits[0]


def moving_mean(values, width=5):
    return [
        sum(values[max(0, i - width + 1) : i + 1]) / len(values[max(0, i - width + 1) : i + 1])
        for i in range(len(values))
    ]


def build_fig2_attribution(th):
    run1 = read_jsonl(DATA / "run1_summary.jsonl")
    baselines = read_jsonl(DATA / "baseline_summary.jsonl")
    final_step = max(row["step"] for row in run1)

    def fetch(cohort, judge):
        records = [
            lookup(baselines, cond="D", judge=judge, set=cohort),
            lookup(baselines, cond="R", judge=judge, set=cohort),
            lookup(run1, step=0, judge=judge, set=cohort),
            lookup(run1, step=final_step, judge=judge, set=cohort),
        ]
        y = np.array([row["rate"] for row in records], dtype=float)
        lo = np.array([row["ci95"][0] for row in records], dtype=float)
        hi = np.array([row["ci95"][1] for row in records], dtype=float)
        return y, lo, hi

    fig = plt.figure(figsize=(th.WIDTH_IN, 2.52))
    gs = fig.add_gridspec(1, 2, wspace=0.21, left=0.075, right=0.985, top=0.77, bottom=0.19)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    xs = np.array([0.0, 1.0, 2.0, 3.0])
    dx = 0.085
    xlabels = ["GPT\ndraft", "Prompted\nOpus", "Supervised\nwriter", "Adversarial\nwriter"]

    configs = [
        (
            ax_a,
            "clean2026",
            f"{th.bold('(a) 2026 working papers')} (137 paragraphs)",
            (-4.5, 89.0),
            [0, 20, 40, 60, 80],
            th.pct("Taken for the authors (%)"),
            True,
        ),
        (
            ax_b,
            "pre2023",
            th.pct(f"{th.bold('(b) 1999--2022 papers')} (54 paragraphs; 0--55% scale)")
            if th.use_pgf
            else "(b) 1999\u20132022 papers (54 paragraphs; 0\u201355% scale)",
            (-3.0, 58.0),
            [0, 10, 20, 30, 40, 50],
            None,
            False,
        ),
    ]

    for ax, cohort, title, ylim, yticks, ylabel, is_left in configs:
        yg, log, hig = fetch(cohort, GPT)
        yo, loo, hio = fetch(cohort, OPUS)

        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        ax.axvspan(1.5, 3.52, color=th.PALE, zorder=0, lw=0)
        ax.axvline(1.5, color=th.LIGHT, lw=0.65, ls="-", zorder=1.0)
        ax.axhline(50.0, color=th.MID, lw=0.70, ls=(0, (2.2, 2.0)), zorder=1.5)
        if is_left:
            ax.text(-0.46, 51.4, th.pct("50% chance"), ha="left", va="bottom", fontsize=7.2, color=th.DARK)
        else:
            ax.text(-0.46, 48.2, th.pct("50% chance"), ha="left", va="top", fontsize=7.2, color=th.DARK)

        top_y = ylim[1] - (1.8 if is_left else 1.2)
        ax.text(0.5, top_y, "Prompted", ha="center", va="top", fontsize=7.2, color=th.DARK)
        ax.text(2.5, top_y, "Trained open model", ha="center", va="top", fontsize=7.2, color=th.BLACK)

        ax.plot(
            [xs[2] - dx, xs[3] - dx],
            [yg[2], yg[3]],
            color=th.BLACK,
            lw=1.25,
            ls="-",
            zorder=2.8,
            path_effects=th.halo(2.0),
        )
        ax.plot(
            [xs[2] + dx, xs[3] + dx],
            [yo[2], yo[3]],
            color=th.DARK,
            lw=1.10,
            ls=(0, (4.2, 2.1)),
            zorder=2.6,
            path_effects=th.halo(1.9),
        )

        for i in range(4):
            xg_i = xs[i] - dx
            xo_i = xs[i] + dx
            if hig[i] > log[i]:
                ax.plot([xg_i, xg_i], [log[i], hig[i]], color=th.BLACK, lw=0.95, solid_capstyle="butt", zorder=3.0)
            if hio[i] > loo[i]:
                ax.plot([xo_i, xo_i], [loo[i], hio[i]], color=th.DARK, lw=0.85, solid_capstyle="butt", zorder=2.9)

        ax.plot(xs - dx, yg, ls="none", marker="o", ms=4.8, mfc=th.BLACK, mec="white", mew=0.65, zorder=4.0)
        ax.plot(xs + dx, yo, ls="none", marker="s", ms=4.4, mfc="white", mec=th.DARK, mew=0.95, zorder=3.9)

        weight_bold = "normal" if th.use_pgf else "bold"
        if is_left:
            ax.text(xs[0] - dx - 0.04, hig[0] + 1.8, f"{yg[0]:.1f}", ha="right", va="bottom", fontsize=7.0, color=th.BLACK)
            ax.text(xs[0] + dx + 0.04, hio[0] + 1.8, f"{yo[0]:.1f}", ha="left", va="bottom", fontsize=7.0, color=th.DARK)
            ax.text(xs[1] - dx, hig[1] + 1.8, f"{yg[1]:.1f}", ha="center", va="bottom", fontsize=7.0, color=th.BLACK)
            ax.text(xs[1] + dx + 0.08, yo[1], f"{yo[1]:.1f}", ha="left", va="center", fontsize=7.0, color=th.DARK)
            ax.text(xs[2] - dx - 0.06, yg[2] + 0.5, f"{yg[2]:.1f}", ha="right", va="center", fontsize=7.0, color=th.BLACK)
            ax.text(xs[2] + dx + 0.07, yo[2] - 1.2, f"{yo[2]:.1f}", ha="left", va="center", fontsize=7.0, color=th.DARK)
            ax.text(xs[3] - dx - 0.06, yg[3] + 1.0, th.bold("70.8"), ha="right", va="center", fontsize=7.3, fontweight=weight_bold, color=th.BLACK)
            ax.text(xs[3] + dx + 0.07, yo[3], th.bold("55.8"), ha="left", va="center", fontsize=7.2, fontweight=weight_bold, color=th.DARK)
        else:
            ax.text(xs[0] - dx - 0.04, 1.5, f"{yg[0]:.1f}", ha="right", va="bottom", fontsize=7.0, color=th.BLACK)
            ax.text(xs[0] + dx + 0.04, 1.5, f"{yo[0]:.1f}", ha="left", va="bottom", fontsize=7.0, color=th.DARK)
            ax.text(xs[1] - dx - 0.03, hig[1] + 1.2, f"{yg[1]:.1f}", ha="right", va="bottom", fontsize=7.0, color=th.BLACK)
            ax.text(xs[1] + dx + 0.03, hio[1] + 1.2, f"{yo[1]:.1f}", ha="left", va="bottom", fontsize=7.0, color=th.DARK)
            ax.text(xs[2] - dx - 0.06, yg[2] - 0.5, f"{yg[2]:.1f}", ha="right", va="center", fontsize=7.0, color=th.BLACK)
            ax.text(xs[2] + dx + 0.08, yo[2] - 1.6, f"{yo[2]:.1f}", ha="left", va="top", fontsize=7.0, color=th.DARK)
            ax.text(xs[3] - dx - 0.06, yg[3] + 0.8, th.bold("31.5"), ha="right", va="center", fontsize=7.3, fontweight=weight_bold, color=th.BLACK)
            ax.text(xs[3] + dx + 0.08, yo[3] - 0.6, th.bold("25.9"), ha="left", va="center", fontsize=7.2, fontweight=weight_bold, color=th.DARK)

        ax.set_xlim(-0.52, 3.52)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=7.6)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels([th.pct(f"{tick}%") for tick in yticks])
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=4)
        ax.text(0.0, 1.05, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.3)

    leg = fig.add_axes([0.075, 0.885, 0.91, 0.09])
    leg.axis("off")
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    leg.plot([0.16, 0.215], [0.5, 0.5], color=th.BLACK, lw=1.25, ls="-")
    leg.plot(0.1875, 0.5, marker="o", ms=4.8, mfc=th.BLACK, mec="white", mew=0.65)
    leg.text(0.228, 0.5, th.pct("GPT-6.1 Sol judge (95% CI)"), va="center", ha="left", fontsize=7.9, color=th.BLACK)

    leg.plot([0.55, 0.605], [0.5, 0.5], color=th.DARK, lw=1.10, ls=(0, (4.2, 2.1)))
    leg.plot(0.5775, 0.5, marker="s", ms=4.4, mfc="white", mec=th.DARK, mew=0.95)
    leg.text(0.618, 0.5, th.pct("Claude Opus 5.5 judge (95% CI)"), va="center", ha="left", fontsize=7.9, color=th.DARK)

    th.export(fig, FIGURES / "fig2_attribution")
    plt.close(fig)


def build_figA1_training(th):
    logs = {
        1: read_jsonl(DATA / "training_log_run1.jsonl"),
        2: read_jsonl(DATA / "training_log_run2.jsonl"),
    }
    pause_step = min(row["step"] for log in logs.values() for row in log if row["paused"])

    s1 = np.array([row["step"] for row in logs[1]], dtype=float)
    a1 = np.array([row["judge_ema"] for row in logs[1]], dtype=float)
    p1 = np.array(moving_mean([row["reward_p"] for row in logs[1]]), dtype=float)
    s2 = np.array([row["step"] for row in logs[2]], dtype=float)
    a2 = np.array([row["judge_ema"] for row in logs[2]], dtype=float)
    p2 = np.array(moving_mean([row["reward_p"] for row in logs[2]]), dtype=float)

    fig = plt.figure(figsize=(th.WIDTH_IN, 2.55))
    gs = fig.add_gridspec(1, 2, wspace=0.22, left=0.08, right=0.985, top=0.76, bottom=0.18)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    ax_a.yaxis.grid(True)
    ax_a.xaxis.grid(False)
    ax_a.axvspan(pause_step, 75, color=th.PALE, zorder=0, lw=0)
    ax_a.axvline(pause_step, color=th.LIGHT, lw=0.6, zorder=1.0)
    ax_a.axhline(0.60, color=th.MID, lw=0.7, ls=(0, (2.2, 2.0)), zorder=1.5)
    ax_a.text(2.0, 0.585, "Pause threshold (0.60)", ha="left", va="top", fontsize=7.2, color=th.DARK)
    ax_a.text(65.0, 0.975, "Writer\npaused", ha="center", va="top", fontsize=7.2, color=th.DARK)

    ax_a.plot(s1, a1, color=th.BLACK, lw=1.30, ls="-", zorder=3.5, path_effects=th.halo(2.0))
    ax_a.plot(s2, a2, color=th.DARK, lw=1.15, ls=(0, (4.5, 2.2)), zorder=3.2, path_effects=th.halo(1.9))
    ax_a.plot(s1[-1], a1[-1], marker="o", ms=3.8, mfc=th.BLACK, mec="white", mew=0.6, zorder=4.0)
    ax_a.plot(s2[-1], a2[-1], marker="s", ms=3.6, mfc="white", mec=th.DARK, mew=0.9, zorder=3.9)

    ax_a.set_xlim(0, 75)
    ax_a.set_xticks([0, 25, 50, 75])
    ax_a.set_ylim(0.45, 1.01)
    ax_a.set_yticks([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    ax_a.set_yticklabels([f"{y:.2f}" for y in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]])
    ax_a.set_xlabel("Training step", labelpad=4)
    ax_a.set_ylabel("Accuracy", labelpad=4)
    ax_a.text(
        0.0,
        1.05,
        f"{th.bold('(a) Internal judge accuracy')} (running mean)",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
    )

    ax_b.yaxis.grid(True)
    ax_b.xaxis.grid(False)
    ax_b.axvspan(pause_step, 75, color=th.PALE, zorder=0, lw=0)
    ax_b.axvline(pause_step, color=th.LIGHT, lw=0.6, zorder=1.0)
    ax_b.axhline(0.50, color=th.MID, lw=0.7, ls=(0, (2.2, 2.0)), zorder=1.5)
    ax_b.text(2.0, 0.515, "Chance (0.50)", ha="left", va="bottom", fontsize=7.2, color=th.DARK)
    ax_b.text(65.0, 0.625, "Writer\npaused", ha="center", va="top", fontsize=7.2, color=th.DARK)

    ax_b.plot(s1, p1, color=th.BLACK, lw=1.30, ls="-", zorder=3.5, path_effects=th.halo(2.0))
    ax_b.plot(s2, p2, color=th.DARK, lw=1.15, ls=(0, (4.5, 2.2)), zorder=3.2, path_effects=th.halo(1.9))
    ax_b.plot(s1[-1], p1[-1], marker="o", ms=3.8, mfc=th.BLACK, mec="white", mew=0.6, zorder=4.0)
    ax_b.plot(s2[-1], p2[-1], marker="s", ms=3.6, mfc="white", mec=th.DARK, mew=0.9, zorder=3.9)

    ax_b.set_xlim(0, 75)
    ax_b.set_xticks([0, 25, 50, 75])
    ax_b.set_ylim(0.0, 0.64)
    ax_b.set_yticks([0.00, 0.15, 0.30, 0.45, 0.60])
    ax_b.set_yticklabels([f"{y:.2f}" for y in [0.00, 0.15, 0.30, 0.45, 0.60]])
    ax_b.set_xlabel("Training step", labelpad=4)
    ax_b.set_ylabel("Mean probability", labelpad=4)
    ax_b.text(
        0.0,
        1.05,
        f"{th.bold('(b) Judge probability of authors')} (5-step mean)",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
    )

    leg = fig.add_axes([0.04, 0.88, 0.94, 0.09])
    leg.axis("off")
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    leg.plot([0.07, 0.125], [0.5, 0.5], color=th.BLACK, lw=1.30, ls="-")
    leg.plot(0.0975, 0.5, marker="o", ms=3.8, mfc=th.BLACK, mec="white", mew=0.6)
    leg.text(0.138, 0.5, "Run 1 (seed 20261115; stops step 72)", va="center", ha="left", fontsize=7.9, color=th.BLACK)

    leg.plot([0.55, 0.605], [0.5, 0.5], color=th.DARK, lw=1.15, ls=(0, (4.5, 2.2)))
    leg.plot(0.5775, 0.5, marker="s", ms=3.6, mfc="white", mec=th.DARK, mew=0.9)
    leg.text(0.618, 0.5, "Run 2 (seed 20261116; stops step 69)", va="center", ha="left", fontsize=7.9, color=th.DARK)

    th.export(fig, FIGURES / "figA1_training")
    plt.close(fig)


def build_figA2_readers(th):
    stats = json.loads((TABLES / "reading_test.json").read_text(encoding="utf-8"))
    rows = [
        ("GPT draft", "all_draft", True),
        ("Supervised writer", "all_v1", False),
        ("Adversarial writer", "all_ral", False),
    ]

    fig = plt.figure(figsize=(th.WIDTH_IN, 2.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.08], wspace=0.04, left=0.21, right=0.985, top=0.80, bottom=0.23)
    ax = fig.add_subplot(gs[0])
    ax_tab = fig.add_subplot(gs[1], sharey=ax)

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.plot([50.0, 50.0], [-0.36, 2.55], color=th.BLACK, lw=0.70, ls=(0, (2.2, 2.0)), zorder=1.6)
    ax.text(50.0, -0.48, th.pct("50% chance"), ha="center", va="bottom", fontsize=7.3, color=th.DARK)

    ax.axhspan(0.55, 2.45, color=th.PALE, zorder=0, lw=0)
    ax_tab.axhspan(0.55, 2.45, color=th.PALE, zorder=0, lw=0)
    ax_tab.axis("off")
    ax_tab.set_xlim(0, 1)

    weight_bold = "normal" if th.use_pgf else "bold"
    ax_tab.text(0.22, -0.48, th.bold("Share"), ha="right", va="bottom", fontsize=7.8, fontweight=weight_bold, color=th.BLACK)
    ax_tab.text(0.56, -0.48, th.pct(th.bold("[95% CI]")), ha="center", va="bottom", fontsize=7.8, fontweight=weight_bold, color=th.BLACK)
    ax_tab.text(0.99, -0.48, th.bold("Found/N"), ha="right", va="bottom", fontsize=7.8, fontweight=weight_bold, color=th.BLACK)

    for y, (_, key, is_draft) in enumerate(rows):
        entry = stats[key]
        pt = 100.0 * entry["k"] / entry["n"]
        lo = 100.0 * entry["lo"]
        hi = 100.0 * entry["hi"]
        if is_draft:
            ax.plot([lo, hi], [y, y], color=th.BLACK, lw=1.15, solid_capstyle="butt", zorder=3.0)
            ax.plot(pt, y, marker="o", ms=5.3, mfc=th.BLACK, mec="white", mew=0.65, zorder=4.0)
            ax_tab.text(0.22, y, th.pct(th.bold(f"{pt:.1f}%")), ha="right", va="center", fontsize=8.0, fontweight=weight_bold, color=th.BLACK)
            ax_tab.text(0.56, y, f"[{lo:.1f}, {hi:.1f}]", ha="center", va="center", fontsize=7.8, color=th.BLACK)
            ax_tab.text(0.99, y, f"{entry['k']}/{entry['n']}", ha="right", va="center", fontsize=7.8, color=th.BLACK)
        else:
            ax.plot([lo, hi], [y, y], color=th.DARK, lw=1.05, ls=(0, (4.2, 2.0)), zorder=3.0)
            ax.plot(pt, y, marker="s", ms=4.8, mfc="white", mec=th.DARK, mew=1.0, zorder=4.0)
            ax_tab.text(0.22, y, th.pct(f"{pt:.1f}%"), ha="right", va="center", fontsize=8.0, color=th.DARK)
            ax_tab.text(0.56, y, f"[{lo:.1f}, {hi:.1f}]", ha="center", va="center", fontsize=7.8, color=th.DARK)
            ax_tab.text(0.99, y, f"{entry['k']}/{entry['n']}", ha="right", va="center", fontsize=7.8, color=th.DARK)

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([th.pct(f"{tick}%") for tick in [0, 25, 50, 75, 100]])
    ax.set_xlabel(th.pct("Authors' paragraph identified (%)"), labelpad=4)
    ax.set_ylim(2.55, -0.65)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([th.bold("GPT draft"), "Supervised writer", "Adversarial writer"], fontsize=8.1)

    th.export(fig, FIGURES / "figA2_readers")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    use_pgf = has_lualatex()
    if use_pgf:
        mpl.use("pgf")
    else:
        mpl.use("Agg")
    th = MonochromeTheme(use_pgf=use_pgf)
    with th.context():
        build_fig2_attribution(th)
        build_figA1_training(th)
        build_figA2_readers(th)
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
