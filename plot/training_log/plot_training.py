import os
import json
import argparse
import matplotlib.pyplot as plt


def load_json_metrics(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = ["step", "loss", "ppl"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {json_path}")

    return data


def infer_optimizer_name(json_path):
    file_name = os.path.basename(json_path)
    name = os.path.splitext(file_name)[0]

    # You can customize these replacements according to your file names.
    name = name.replace("_metrics", "")
    name = name.replace("train_", "")
    name = name.replace("log_", "")

    return name


def plot_metric(json_files, metric_name, save_path):
    plt.figure(figsize=(8, 5))

    for json_path in json_files:
        data = load_json_metrics(json_path)

        steps = data["step"]
        values = data[metric_name]
        optimizer_name = infer_optimizer_name(json_path)

        plt.plot(
            steps,
            values,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=optimizer_name
        )

    plt.xlabel("Step")
    plt.ylabel(metric_name.upper() if metric_name == "ppl" else metric_name.capitalize())
    plt.title(f"{metric_name.upper() if metric_name == 'ppl' else metric_name.capitalize()} vs Step")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure to {save_path}")


def collect_json_files(json_dir):
    json_files = []

    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            json_files.append(os.path.join(json_dir, file_name))

    json_files = sorted(json_files)

    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")

    return json_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./figures")
    args = parser.parse_args()

    json_files = collect_json_files(args.json_dir)

    print("Found JSON files:")
    for path in json_files:
        print("  ", path)

    plot_metric(
        json_files=json_files,
        metric_name="loss",
        save_path=os.path.join(args.save_dir, "loss_vs_step.png")
    )

    plot_metric(
        json_files=json_files,
        metric_name="ppl",
        save_path=os.path.join(args.save_dir, "ppl_vs_step.png")
    )


if __name__ == "__main__":
    main()