import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;

mobileone = pd.read_csv("mobileone_s0_time");
mobileone["model"] = "MobileOne-s0";

mobilenet2 = pd.read_csv("mobilenetv2_time");
mobilenet2["model"] = "MobileNet-v2";

resnet18 = pd.read_csv("resnet18_time");
resnet18["model"] = "ResNet18";

df = pd.concat(
    [mobileone, mobilenet2, resnet18],
    ignore_index=True);

sns.set_theme(style="whitegrid");

plt.figure(figsize=(10, 4));

sns.lineplot(
    data=df,
    x="frame",
    y="frame_time",
    hue="model",
    linewidth=1);

plt.ylim(0, 1.0);
plt.xlabel("Frame index");
plt.ylabel("Seconds per frame");
plt.title("Frame Time Comparison");

plt.tight_layout();
plt.savefig("frame_time_comparison.png", dpi=150);
plt.close();

