import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_step(path):
    name = os.path.basename(path)
    return int(name.split("_step_")[-1].replace(".pt", ""))


def get_layer_name(path):
    name = os.path.basename(path)
    return name.split("_step_")[0]


def load_subspace(path):
    data = torch.load(path, map_location="cpu")
    mat = data["ortho_matrix"]

    if isinstance(mat, (list, tuple)):
        mat = mat[0]

    mat = mat.float()

    if mat.dim() != 2:
        raise ValueError(f"Expected a 2D matrix, got {mat.shape} from {path}")

    if mat.shape[0] > mat.shape[1]:
        mat = mat.T

    mat = torch.nn.functional.normalize(mat, dim=1)
    return mat


def subspace_similarity(a, b):
    product = a @ b.T
    similarity = torch.linalg.norm(product, ord="fro").item() ** 2
    similarity = similarity / min(a.shape[0], b.shape[0])
    return similarity


def plot_one_layer(layer_name, paths, output_dir):
    paths = sorted(paths, key=get_step)
    matrices = [load_subspace(path) for path in paths]

    n = len(matrices)
    sim_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = subspace_similarity(matrices[i], matrices[j])

    min_val = float(sim_matrix.min())
    max_val = float(sim_matrix.max())
    mean_val = float(sim_matrix.mean())
    median_val = float(np.median(sim_matrix))
    std_val = float(sim_matrix.std())

    print("=" * 80)
    print(f"Layer: {layer_name}")
    print(f"Num steps : {n}")
    print(f"Min       : {min_val:.6f}")
    print(f"Max       : {max_val:.6f}")
    print(f"Mean      : {mean_val:.6f}")
    print(f"Median    : {median_val:.6f}")
    print(f"Std       : {std_val:.6f}")

    npy_path = os.path.join(output_dir, f"{layer_name}_similarity_matrix.npy")
    np.save(npy_path, sim_matrix)

    txt_path = os.path.join(output_dir, f"{layer_name}_stats.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Layer: {layer_name}\n")
        f.write(f"Num steps : {n}\n")
        f.write(f"Min       : {min_val:.6f}\n")
        f.write(f"Max       : {max_val:.6f}\n")
        f.write(f"Mean      : {mean_val:.6f}\n")
        f.write(f"Median    : {median_val:.6f}\n")
        f.write(f"Std       : {std_val:.6f}\n")

    plt.figure(figsize=(8, 7))

    image = plt.imshow(
        sim_matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )

    colorbar = plt.colorbar(image)
    colorbar.set_label("Subspace Similarity")

    plt.xticks([])
    plt.yticks([])

    plt.xlabel("Training Step")
    plt.ylabel("Training Step")

    plt.title(
        f"{layer_name}\n"
        f"Mean={mean_val:.4f}, Median={median_val:.4f}, Min={min_val:.4f}"
    )

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{layer_name}_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved heatmap : {save_path}")
    print(f"Saved matrix  : {npy_path}")
    print(f"Saved stats   : {txt_path}")


def main():
    subspace_dir = "/home/ycyh/code/LLM/LoRA-main/LoRA-main/examples/NLG/subspace/rank_1_gap_50"
    output_dir = "./figures/subspace_heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(os.path.join(subspace_dir, "*.pt"))

    if len(files) == 0:
        print(f"No .pt files found in {subspace_dir}")
        return

    layer_to_files = {}

    for path in files:
        layer_name = get_layer_name(path)
        layer_to_files.setdefault(layer_name, []).append(path)

    print(f"Found {len(files)} .pt files.")
    print(f"Found {len(layer_to_files)} layers.")

    for layer_name, paths in sorted(layer_to_files.items()):
        if len(paths) < 2:
            print(f"Skip {layer_name}: only {len(paths)} file.")
            continue

        plot_one_layer(layer_name, paths, output_dir)


if __name__ == "__main__":
    main()