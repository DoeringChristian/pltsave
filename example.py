"""Quick example demonstrating pltsave functionality."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pltsave import save_figure, load_figure, compare_figures

# Create a simple figure
print("Creating figure...")
fig, ax = plt.subplots(figsize=(8, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), 'r-', linewidth=2, label='sin(x)')
ax.plot(x, np.cos(x), 'b--', linewidth=2, label='cos(x)')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Trigonometric Functions')
ax.legend()
ax.grid(True, alpha=0.3)

# Save the figure
filename = 'example_figure.json'
print(f"Saving to {filename}...")
save_figure(fig, filename)
print("✓ Saved successfully")

# Load it back
print(f"Loading from {filename}...")
loaded_fig = load_figure(filename)
print("✓ Loaded successfully")

# Compare them
print("Comparing original and loaded figures...")
is_equal, report = compare_figures(fig, loaded_fig)

if is_equal:
    print("✓ Figures are identical!")
    print(f"  - Number of axes: {report['summary']['num_axes']}")
    print(f"  - Number of differences: {report['summary']['num_differences']}")
else:
    print("⚠ Figures have some differences:")
    for diff in report['differences'][:5]:
        print(f"  - {diff}")

# Save both as PNG for visual comparison
print("\nSaving as PNG for visual inspection...")
fig.savefig('example_original.png', dpi=100, bbox_inches='tight')
loaded_fig.savefig('example_loaded.png', dpi=100, bbox_inches='tight')
print("✓ Saved example_original.png and example_loaded.png")

print("\n✓ Example completed successfully!")
print(f"Serialized data saved to: {filename}")

# Clean up
plt.close(fig)
plt.close(loaded_fig)
