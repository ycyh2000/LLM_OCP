import re
import json
import argparse
import os


def parse_training_log(log_path):
    pattern = re.compile(
        r"""
        epoch\s+(?P<epoch>\d+).*?
        step\s+(?P<step>\d+).*?
        (?P<batches>\d+)\s+batches.*?
        lr\s+(?P<lr>[0-9eE+\-.]+).*?
        ms/batch\s+(?P<ms_per_batch>[0-9.]+).*?
        loss\s+(?P<loss>[0-9.]+).*?
        avg\s+loss\s+(?P<avg_loss>[0-9.]+).*?
        ppl\s+(?P<ppl>[0-9.]+)
        """,
        re.VERBOSE
    )

    data = {
        "epoch": [],
        "step": [],
        "batches": [],
        "lr": [],
        "ms_per_batch": [],
        "loss": [],
        "avg_loss": [],
        "ppl": []
    }

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)

            if match is None:
                continue

            item = match.groupdict()

            data["epoch"].append(int(item["epoch"]))
            data["step"].append(int(item["step"]))
            data["batches"].append(int(item["batches"]))
            data["lr"].append(float(item["lr"]))
            data["ms_per_batch"].append(float(item["ms_per_batch"]))
            data["loss"].append(float(item["loss"]))
            data["avg_loss"].append(float(item["avg_loss"]))
            data["ppl"].append(float(item["ppl"]))

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    data = parse_training_log(args.log_path)

    if args.save_path is None:
        base_name = os.path.splitext(os.path.basename(args.log_path))[0]
        args.save_path = base_name + ".json"

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Saved {len(data['step'])} records to {args.save_path}")


if __name__ == "__main__":
    main()