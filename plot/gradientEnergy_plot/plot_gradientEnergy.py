import torch
import sys

def inspect_pt_file(filepath):
    """
    Load a .pt file and display its contents.
    """
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)

        print(f"\nFile: {filepath}")
        print(f"Top-level type: {type(data)}")
        print("-" * 50)

        if isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys: {list(data.keys())}\n")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  Key '{key}': Tensor, shape {value.shape}, dtype {value.dtype}")
                elif isinstance(value, dict):
                    print(f"  Key '{key}': nested dict ({len(value)} subkeys)")
                elif isinstance(value, (list, tuple)):
                    print(f"  Key '{key}': list/tuple (length {len(value)})")
                else:
                    print(f"  Key '{key}': {type(value).__name__} -> {value}")

        elif isinstance(data, torch.nn.Module):
            print(f"Complete model object (nn.Module):\n{data}")

        elif isinstance(data, torch.Tensor):
            print(f"Tensor, shape {data.shape}, dtype {data.dtype}")
            if data.numel() <= 20:
                print(f"Values: {data.tolist()}")
            else:
                print(f"First 5 values: {data.flatten()[:5].tolist()} ...")

        elif isinstance(data, (list, tuple)):
            print(f"List/tuple of length {len(data)}")
            for i, item in enumerate(data):
                if isinstance(item, torch.Tensor):
                    print(f"  Index [{i}]: Tensor, shape {item.shape}")
                else:
                    print(f"  Index [{i}]: {type(item).__name__} -> {item}")

        else:
            print(f"Content: {data}")

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_pt_file(sys.argv[1])
    else:
        print("Usage: python script.py <path_to_pt_file>")
        # To test directly, uncomment and set your path:
        # inspect_pt_file("model.pt")