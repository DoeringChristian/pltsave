"""
Comprehensive tests for pltsave library.

Tests cover all major matplotlib features including:
- Line plots
- Scatter plots
- Bar charts
- Histograms
- Images
- Patches (shapes)
- Annotations
- Legends
- Colorbars
- 3D plots
- Multiple subplots
- And more!
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pltsave import save_figure, load_figure, compare_figures


class TestBasicPlots:
    """Test basic plot types."""

    def test_simple_line_plot(self, temp_file):
        """Test saving and loading a simple line plot."""
        # Create original figure
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, color="blue", linewidth=2, label="sin(x)")
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_title("Simple Line Plot")
        ax.legend()

        # Save and load
        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        # Compare
        is_equal, report = compare_figures(fig, loaded_fig)
        if not is_equal:
            print("Differences found:")
            for diff in report["differences"]:
                print(f"  - {diff}")

        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_multiple_lines(self, temp_file):
        """Test multiple lines with different styles."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)

        ax.plot(x, np.sin(x), "r-", linewidth=2, label="sin")
        ax.plot(x, np.cos(x), "b--", linewidth=2, label="cos")
        ax.plot(x, np.tan(x), "g:", linewidth=2, label="tan")

        ax.set_xlim(0, 10)
        ax.set_ylim(-2, 2)
        ax.legend()

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_markers_and_styles(self, temp_file):
        """Test different markers and line styles."""
        fig, ax = plt.subplots()
        x = np.arange(10)
        y = x**2

        ax.plot(
            x,
            y,
            marker="o",
            markersize=8,
            markerfacecolor="red",
            markeredgecolor="black",
            markeredgewidth=2,
            linestyle="-",
            linewidth=2,
            color="blue",
        )

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


class TestScatterPlots:
    """Test scatter plots."""

    def test_simple_scatter(self, temp_file):
        """Test a simple scatter plot."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)

        ax.scatter(x, y, s=50, c="red", alpha=0.5, marker="o")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Scatter Plot")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_scatter_with_colors(self, temp_file):
        """Test scatter plot with color mapping."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)
        colors = np.random.rand(100)
        sizes = 100 * np.random.rand(100)

        ax.scatter(x, y, s=sizes, c=colors, alpha=0.5, cmap="viridis")
        ax.set_title("Scatter with Colors")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


class TestBarCharts:
    """Test bar charts."""

    def test_simple_bar_chart(self, temp_file):
        """Test a simple bar chart."""
        fig, ax = plt.subplots()
        categories = ["A", "B", "C", "D"]
        values = [3, 7, 2, 5]

        ax.bar(categories, values, color="skyblue", edgecolor="black", linewidth=1.5)
        ax.set_xlabel("Category")
        ax.set_ylabel("Value")
        ax.set_title("Bar Chart")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_horizontal_bar_chart(self, temp_file):
        """Test a horizontal bar chart."""
        fig, ax = plt.subplots()
        categories = ["A", "B", "C", "D"]
        values = [3, 7, 2, 5]

        ax.barh(categories, values, color="coral")
        ax.set_xlabel("Value")
        ax.set_ylabel("Category")
        ax.set_title("Horizontal Bar Chart")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


class TestHistograms:
    """Test histograms."""

    def test_simple_histogram(self, temp_file):
        """Test a simple histogram."""
        fig, ax = plt.subplots()
        data = np.random.randn(1000)

        ax.hist(data, bins=30, color="green", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


class TestPatches:
    """Test patches (shapes)."""

    def test_rectangles(self, temp_file):
        """Test rectangles."""
        fig, ax = plt.subplots()

        from matplotlib.patches import Rectangle

        rect1 = Rectangle(
            (0.2, 0.2), 0.3, 0.4, facecolor="blue", edgecolor="red", linewidth=2
        )
        rect2 = Rectangle((0.6, 0.3), 0.2, 0.5, facecolor="green", alpha=0.5)

        ax.add_patch(rect1)
        ax.add_patch(rect2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("Rectangles")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_circles(self, temp_file):
        """Test circles."""
        fig, ax = plt.subplots()

        from matplotlib.patches import Circle

        circle1 = Circle(
            (0.3, 0.5), 0.2, facecolor="red", edgecolor="black", linewidth=2
        )
        circle2 = Circle((0.7, 0.5), 0.15, facecolor="blue", alpha=0.5)

        ax.add_patch(circle1)
        ax.add_patch(circle2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("Circles")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_polygons(self, temp_file):
        """Test polygons."""
        fig, ax = plt.subplots()

        from matplotlib.patches import Polygon

        triangle = Polygon(
            [[0.2, 0.2], [0.5, 0.8], [0.8, 0.2]],
            facecolor="yellow",
            edgecolor="red",
            linewidth=2,
        )

        ax.add_patch(triangle)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("Polygon")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_wedge(self, temp_file):
        """Test wedge (pie slice)."""
        fig, ax = plt.subplots()

        from matplotlib.patches import Wedge

        wedge = Wedge(
            (0.5, 0.5), 0.3, 30, 120, facecolor="orange", edgecolor="black", linewidth=2
        )

        ax.add_patch(wedge)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("Wedge")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


class TestAnnotations:
    """Test text and annotations."""

    def test_simple_text(self, temp_file):
        """Test simple text."""
        fig, ax = plt.subplots()

        ax.plot([0, 1], [0, 1])
        ax.text(
            0.5,
            0.5,
            "Center Text",
            fontsize=14,
            ha="center",
            va="center",
            color="red",
            rotation=45,
            weight="bold",
        )

        ax.set_title("Text Example")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_annotations(self, temp_file):
        """Test annotations with arrows."""
        fig, ax = plt.subplots()

        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)

        ax.annotate(
            "Peak",
            xy=(np.pi / 2, 1),
            xytext=(np.pi / 2 + 1, 0.5),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
            fontsize=12,
            color="red",
        )

        ax.set_title("Annotation Example")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


class TestImages:
    """Test image display."""

    def test_imshow(self, temp_file):
        """Test imshow with random data."""
        fig, ax = plt.subplots()

        data = np.random.rand(10, 10)
        im = ax.imshow(data, cmap="viridis", interpolation="nearest")

        ax.set_title("Image Display")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_imshow_with_extent(self, temp_file):
        """Test imshow with custom extent."""
        fig, ax = plt.subplots()

        data = np.random.rand(20, 30)
        im = ax.imshow(data, extent=[0, 3, 0, 2], cmap="plasma", alpha=0.8)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Image with Extent")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


class TestSubplots:
    """Test multiple subplots."""

    def test_two_subplots(self, temp_file):
        """Test figure with two subplots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # First subplot
        x = np.linspace(0, 10, 100)
        ax1.plot(x, np.sin(x), "r-")
        ax1.set_title("Sine")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        # Second subplot
        ax2.plot(x, np.cos(x), "b-")
        ax2.set_title("Cosine")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")

        fig.suptitle("Two Subplots", fontsize=16, fontweight="bold")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_grid_subplots(self, temp_file):
        """Test 2x2 grid of subplots."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        x = np.linspace(0, 10, 100)

        axes[0, 0].plot(x, x, "r-")
        axes[0, 0].set_title("Linear")

        axes[0, 1].plot(x, x**2, "g-")
        axes[0, 1].set_title("Quadratic")

        axes[1, 0].plot(x, np.sin(x), "b-")
        axes[1, 0].set_title("Sine")

        axes[1, 1].plot(x, np.exp(-x / 10), "m-")
        axes[1, 1].set_title("Exponential")

        fig.tight_layout()

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


class TestAxisProperties:
    """Test various axis properties."""

    def test_log_scale(self, temp_file):
        """Test logarithmic scale."""
        fig, ax = plt.subplots()

        x = np.logspace(0, 3, 100)
        y = x**2

        ax.plot(x, y)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("X (log scale)")
        ax.set_ylabel("Y (log scale)")
        ax.set_title("Log-Log Plot")
        ax.grid(True)

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_custom_ticks(self, temp_file):
        """Test custom tick positions and labels."""
        fig, ax = plt.subplots()

        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x))

        ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi])
        ax.set_xticklabels(["0", "π", "2π", "3π"])

        ax.set_title("Custom Ticks")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_aspect_ratio(self, temp_file):
        """Test aspect ratio."""
        fig, ax = plt.subplots()

        ax.plot([0, 1], [0, 1])
        ax.set_aspect("equal")
        ax.set_title("Equal Aspect Ratio")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


class Test3DPlots:
    """Test 3D plots."""

    def test_3d_line_plot(self, temp_file):
        """Test 3D line plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        t = np.linspace(0, 10, 100)
        x = np.sin(t)
        y = np.cos(t)
        z = t

        ax.plot(x, y, z, "r-", linewidth=2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Line Plot")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_3d_scatter(self, temp_file):
        """Test 3D scatter plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)
        z = np.random.randn(n)
        colors = np.random.rand(n)

        ax.scatter(x, y, z, c=colors, marker="o", s=50)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Scatter Plot")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_3d_surface(self, temp_file):
        """Test 3D surface plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        x = np.linspace(-5, 5, 30)
        y = np.linspace(-5, 5, 30)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))

        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Surface Plot")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


class TestComplexFigures:
    """Test complex figures with multiple features."""

    def test_everything_combined(self, temp_file):
        """Test a figure with many different features."""
        fig = plt.figure(figsize=(12, 8))

        # Main plot
        ax1 = plt.subplot(2, 2, 1)
        x = np.linspace(0, 10, 100)
        ax1.plot(x, np.sin(x), "r-", linewidth=2, label="sin")
        ax1.plot(x, np.cos(x), "b--", linewidth=2, label="cos")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title("Trigonometric Functions")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter plot
        ax2 = plt.subplot(2, 2, 2)
        x_scatter = np.random.randn(50)
        y_scatter = np.random.randn(50)
        colors = np.random.rand(50)
        ax2.scatter(x_scatter, y_scatter, c=colors, s=100, alpha=0.6, cmap="viridis")
        ax2.set_title("Scatter Plot")

        # Bar chart
        ax3 = plt.subplot(2, 2, 3)
        categories = ["A", "B", "C", "D", "E"]
        values = [23, 45, 56, 78, 32]
        ax3.bar(categories, values, color="coral", edgecolor="black", linewidth=1.5)
        ax3.set_title("Bar Chart")
        ax3.set_ylabel("Values")

        # Image
        ax4 = plt.subplot(2, 2, 4)
        data = np.random.rand(20, 20)
        im = ax4.imshow(data, cmap="plasma", interpolation="bilinear")
        ax4.set_title("Heatmap")

        fig.suptitle("Combined Features Test", fontsize=16, fontweight="bold")
        fig.tight_layout()

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_figure(self, temp_file):
        """Test empty figure."""
        fig = plt.figure()

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_figure_with_empty_axes(self, temp_file):
        """Test figure with axes but no data."""
        fig, ax = plt.subplots()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Empty Axes")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)

    def test_invisible_elements(self, temp_file):
        """Test invisible elements."""
        fig, ax = plt.subplots()

        line1 = ax.plot([0, 1], [0, 1], "r-", label="visible")[0]
        line2 = ax.plot([0, 1], [1, 0], "b-", label="invisible")[0]
        line2.set_visible(False)

        ax.set_title("Invisible Elements")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"

        plt.close(fig)
        plt.close(loaded_fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
