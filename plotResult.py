import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ====== Configuration ======
file_path = "result.xlsx"          # Replace with your file path
output_figure = "training_curves.png"  # Output image file name
start_epoch = 0                    # Epoch number corresponding to the first row of data (0 or 1)
# ============================

# Read all sheets
all_sheets = pd.read_excel(file_path, sheet_name=None)

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes.flatten()

# Define the four metrics to plot and their y-axis labels
# Format: (keyword that may appear in Excel column names, y-axis display label)
metrics = [
    ("train loss", "Train Loss"),
    ("validation loss", "Validation Loss"),
    ("train ppl", "Train Perplexity"),
    ("validation ppl", "Validation Perplexity")
]


allowed_sheets = {'0.0001adamAndSignOcp', '0.0001adamAndSignOcpWithNorm'}




# Iterate over each sheet (each sheet represents one algorithm)
for sheet_name, df in all_sheets.items():
    if sheet_name not in allowed_sheets:
        continue
    print(sheet_name)
    # X-axis: row numbers (starting from 0), offset by start_epoch
    epochs = range(start_epoch, start_epoch + len(df))

    for i, (col_keyword, ylabel) in enumerate(metrics):
        # Search for a column containing col_keyword (case-insensitive, ignoring spaces)
        matched_col = None
        for col in df.columns:
            # Normalize column name: lowercase, remove spaces
            col_norm = str(col).lower().replace(" ", "")
            keyword_norm = col_keyword.lower().replace(" ", "")
            if col_norm == keyword_norm:
                matched_col = col
                break
        if matched_col is None:
            print(f"Warning: Column '{col_keyword}' not found in sheet '{sheet_name}'. Skipping this metric.")
            continue

        # Plot the curve for this algorithm
        ax[i].plot(epochs, df[matched_col], marker='o', label=sheet_name)

# Format each subplot
for i, (_, ylabel) in enumerate(metrics):
    ax[i].set_xlabel("Epoch")
    ax[i].set_ylabel(ylabel)
    ax[i].legend(fontsize=8)
    ax[i].grid(True, linestyle='--', alpha=0.6)
    # Force integer ticks on the x-axis
    ax[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

fig.suptitle("Training Curves Across Algorithms", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_figure, dpi=300, bbox_inches='tight')
plt.show()

print(f"Figure saved as {output_figure}")