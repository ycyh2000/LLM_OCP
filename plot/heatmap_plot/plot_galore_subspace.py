import os
import glob
import torch
import matplotlib.pyplot as plt


def load_ortho_matrix(path):
    data = torch.load(path, map_location="cpu")
    mat = data["ortho_matrix"]

    if isinstance(mat, list):
        mat = mat[0]

    return mat.float(), data


def subspace_similarity(A, B):
    """
    Compute subspace similarity between two orthogonal matrices.
    Larger value means more similar.
    """
    if A.dim() != 2 or B.dim() != 2:
        return None

    # Ensure shape is [rank, dim] or [dim, rank]
    if A.shape[0] > A.shape[1]:
        A = A.T
    if B.shape[0] > B.shape[1]:
        B = B.T

    M = A @ B.T
    return torch.linalg.norm(M, ord="fro").item() ** 2 / min(A.shape[0], B.shape[0])


def parse_step(path):
    name = os.path.basename(path)
    return int(name.split("_step_")[-1].replace(".pt", ""))


def main():
    subspace_dir = "/examples/NLG/subspace/rank_1_gap_50"
    files = sorted(glob.glob(os.path.join(subspace_dir, "*.pt")), key=parse_step)

    if len(files) == 0:
        print("No .pt files found.")
        return

    layer_groups = {}

    for f in files:
        base = os.path.basename(f)
        layer_name = base.split("_step_")[0]
        layer_groups.setdefault(layer_name, []).append(f)

    os.makedirs("figures", exist_ok=True)

    for layer_name, paths in layer_groups.items():
        paths = sorted(paths, key=parse_step)

        if len(paths) < 2:
            continue

        ref_matrix, _ = load_ortho_matrix(paths[0])
        steps = []
        sims = []

        for p in paths:
            mat, data = load_ortho_matrix(p)
            sim = subspace_similarity(ref_matrix, mat)

            if sim is None:
                continue

            steps.append(parse_step(p))
            sims.append(sim)

        plt.figure()
        plt.plot(steps, sims, marker="o")
        plt.xlabel("Training Step")
        plt.ylabel("Subspace Similarity to Step 0")
        plt.title(layer_name)
        plt.grid(True)

        save_path = f"figures/{layer_name}_subspace_similarity.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved:", save_path)


if __name__ == "__main__":
    main()