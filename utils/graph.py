import seaborn as sns;
import matplotlib.pyplot as plt;

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

