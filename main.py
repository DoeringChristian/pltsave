"""
Demo script for pltsave library.

This demonstrates saving and loading matplotlib figures with full fidelity.
"""

import numpy as np
import matplotlib.pyplot as plt
from pltsave import save_figure, load_figure, compare_figures


def create_demo_figure():
    """Create a comprehensive demo figure with multiple plot types."""
    fig = plt.figure(figsize=(12, 8))

    # Line plot with multiple lines
    ax1 = plt.subplot(2, 2, 1)
    x = np.linspace(0, 10, 100)
    ax1.plot(x, np.sin(x), 'r-', linewidth=2, label='sin(x)')
    ax1.plot(x, np.cos(x), 'b--', linewidth=2, label='cos(x)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Trigonometric Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot with colors
    ax2 = plt.subplot(2, 2, 2)
    n = 100
    x_scatter = np.random.randn(n)
    y_scatter = np.random.randn(n)
    colors = np.sqrt(x_scatter**2 + y_scatter**2)
    sizes = 100 * np.random.rand(n)
    sc = ax2.scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.6, cmap='viridis')
    ax2.set_title('Scatter Plot with Colors')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Bar chart
    ax3 = plt.subplot(2, 2, 3)
    categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    values = np.random.randint(10, 100, 5)
    bars = ax3.bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'],
                   edgecolor='black', linewidth=1.5)
    ax3.set_title('Bar Chart')
    ax3.set_ylabel('Values')
    ax3.set_xticklabels(categories, rotation=45, ha='right')

    # Heatmap
    ax4 = plt.subplot(2, 2, 4)
    data = np.random.randn(20, 20)
    im = ax4.imshow(data, cmap='coolwarm', interpolation='bilinear', aspect='auto')
    ax4.set_title('Heatmap')
    plt.colorbar(im, ax=ax4)

    # Add figure title
    fig.suptitle('pltsave Demo: Comprehensive Figure Example', fontsize=16, fontweight='bold')
    fig.tight_layout()

    return fig


def main():
    """Run the demo."""
    print("=" * 60)
    print("pltsave Demo - Matplotlib Figure Serialization")
    print("=" * 60)
    print()

    # Create the original figure
    print("1. Creating demo figure with multiple plot types...")
    original_fig = create_demo_figure()
    print("   ✓ Original figure created")
    print()

    # Save the figure
    filename = 'demo_figure.json'
    print(f"2. Saving figure to '{filename}'...")
    save_figure(original_fig, filename)
    print(f"   ✓ Figure saved successfully")
    print()

    # Load the figure
    print(f"3. Loading figure from '{filename}'...")
    loaded_fig = load_figure(filename)
    print("   ✓ Figure loaded successfully")
    print()

    # Compare the figures
    print("4. Comparing original and loaded figures...")
    is_equal, report = compare_figures(original_fig, loaded_fig)

    if is_equal:
        print("   ✓ Figures are identical!")
        print(f"   - Number of axes: {report['summary']['num_axes']}")
        print(f"   - Number of differences: {report['summary']['num_differences']}")
    else:
        print("   ✗ Figures differ:")
        for diff in report['differences'][:5]:  # Show first 5 differences
            print(f"     - {diff}")
        if len(report['differences']) > 5:
            print(f"     ... and {len(report['differences']) - 5} more differences")
    print()

    # Show the figures
    print("5. Displaying figures...")
    print("   Original figure will be shown first.")
    print("   Close the window to see the loaded figure.")
    print()

    # Show original
    original_fig.canvas.manager.set_window_title('Original Figure')
    plt.figure(original_fig.number)
    plt.show(block=True)

    # Show loaded
    loaded_fig.canvas.manager.set_window_title('Loaded Figure')
    plt.figure(loaded_fig.number)
    plt.show(block=True)

    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
