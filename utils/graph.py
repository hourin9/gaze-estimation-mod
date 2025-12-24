import seaborn as sns;
import matplotlib.pyplot as plt;

import csv;

def export_frame_time_plot(frame_times, title, filename):
    sns.set_theme(style="whitegrid");

    plt.figure(figsize=(10, 4));
    sns.lineplot(
        x=range(len(frame_times)),
        y=frame_times,
        linewidth=1);

    plt.xlabel("Frame");
    plt.ylabel("Seconds per frame");
    plt.title(title);

    plt.tight_layout();
    plt.savefig(filename);
    plt.close();

def save_frame_times(filename, frame_times):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "frame_time"])
        for i, t in enumerate(frame_times):
            writer.writerow([i, t])

def save_confidence_log(filename, records):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "pid", "confidence", "pitch", "yaw"]
        )
        writer.writeheader()
        writer.writerows(records)

