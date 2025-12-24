import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("confidence.csv")

sns.set_theme(style="whitegrid")

persons = df["pid"].unique()
n = len(persons)

fig, axes = plt.subplots(
    nrows=n,
    ncols=1,
    figsize=(12, 3 * n),
    sharex=True
)

# Ensure axes is iterable for single-person case
if n == 1:
    axes = [axes]

for ax, pid in zip(axes, persons):
    person_df = df[df["pid"] == pid]

    # ----- Left axis: pitch & yaw -----
    sns.lineplot(
        data=person_df,
        x="frame",
        y="pitch",
        ax=ax,
        label="Pitch",
        color="tab:blue",
        linewidth=1.0
    )

    sns.lineplot(
        data=person_df,
        x="frame",
        y="yaw",
        ax=ax,
        label="Yaw",
        color="tab:orange",
        linewidth=1.0
    )

    ax.set_ylabel("Pitch / Yaw (deg)")
    ax.set_ylim(0, 90.5)

    # ----- Right axis: confidence -----
    ax2 = ax.twinx()

    sns.lineplot(
        data=person_df,
        x="frame",
        y="confidence",
        ax=ax2,
        label="Confidence",
        color="tab:red",
        linewidth=1
    )

    ax2.set_ylabel("Confidence")
    ax2.set_ylim(0, 1.1)

    ax2.axhline(
        0.3,
        color="orange",
        linestyle="--",
        linewidth=1,
        alpha=0.8
    )

    ax2.axhline(
        0.75,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.8
    )

    ax2.text(
        0.01,
        0.3,
        "Suspected of cheating",
        color="orange",
        fontsize=9,
        va="bottom",
        ha="left",
        transform=ax2.get_yaxis_transform(),
    )

    ax2.text(
        0.01,
        0.75,
        "Cheating",
        color="red",
        fontsize=9,
        va="bottom",
        ha="left",
        transform=ax2.get_yaxis_transform(),
    )

    # ----- Title -----
    ax.set_title(pid)

    # ----- Legends (merge both axes) -----
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right"
    )

    # Remove duplicate legends
    ax2.legend_.remove()

axes[-1].set_xlabel("Frame")

plt.tight_layout()
plt.savefig("per_person_pitch_yaw_confidence.png", dpi=150)
plt.close()

