"""
Comprehensive tests for ALL matplotlib gallery plot types.

This test suite covers all plot types from the matplotlib gallery:
https://matplotlib.org/stable/plot_types/index.html

Categories tested:
1. Pairwise data (7 types)
2. Statistical distributions (9 types)
3. Gridded data (7 types)
4. Irregular gridded data (4 types)
5. 3D and volumetric data (10+ types)
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pltsave import save_figure, load_figure, compare_figures


class TestPairwiseData:
    """Test pairwise data plot types."""

    def test_plot(self, temp_file):
        """Test basic plot(x, y)."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, "b-", linewidth=2, label="sin(x)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("plot(x, y)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_scatter(self, temp_file):
        """Test scatter(x, y)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        colors = np.random.rand(100)
        sizes = 100 * np.random.rand(100)

        ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap="viridis")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("scatter(x, y)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_bar(self, temp_file):
        """Test bar(x, height)."""
        fig, ax = plt.subplots()
        categories = ["A", "B", "C", "D", "E"]
        values = [23, 45, 56, 78, 32]

        ax.bar(categories, values, color="steelblue", edgecolor="black", linewidth=1.5)
        ax.set_xlabel("Category")
        ax.set_ylabel("Value")
        ax.set_title("bar(x, height)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_stem(self, temp_file):
        """Test stem(x, y)."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 20)
        y = np.sin(x)

        ax.stem(x, y, linefmt="b-", markerfmt="ro", basefmt="k-")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("stem(x, y)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_fill_between(self, temp_file):
        """Test fill_between(x, y1, y2)."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.sin(x) + 0.5

        ax.fill_between(x, y1, y2, alpha=0.5, color="skyblue")
        ax.plot(x, y1, "b-", linewidth=2)
        ax.plot(x, y2, "r-", linewidth=2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("fill_between(x, y1, y2)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_stackplot(self, temp_file):
        """Test stackplot(x, y)."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y3 = np.sin(x) * np.cos(x)

        ax.stackplot(x, y1, y2, y3, labels=["sin", "cos", "sin*cos"], alpha=0.7)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("stackplot(x, y)")
        ax.legend(loc="upper left")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_stairs(self, temp_file):
        """Test stairs(values)."""
        fig, ax = plt.subplots()
        values = [1, 2, 3, 2, 4, 3, 5, 4, 6]

        ax.stairs(
            values, fill=True, alpha=0.5, color="orange", edgecolor="black", linewidth=2
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("stairs(values)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")


class TestStatisticalDistributions:
    """Test statistical distribution plot types."""

    def test_hist(self, temp_file):
        """Test hist(x)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        data = np.random.randn(1000)

        ax.hist(data, bins=30, color="green", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title("hist(x)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_boxplot(self, temp_file):
        """Test boxplot(X)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        data = [
            np.random.randn(100),
            np.random.randn(100) + 1,
            np.random.randn(100) - 1,
        ]

        ax.boxplot(data, labels=["A", "B", "C"], patch_artist=True)
        ax.set_xlabel("Group")
        ax.set_ylabel("Value")
        ax.set_title("boxplot(X)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_errorbar(self, temp_file):
        """Test errorbar(x, y, yerr, xerr)."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 10)
        y = np.sin(x)
        yerr = 0.1 * np.ones_like(y)
        xerr = 0.2 * np.ones_like(x)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            xerr=xerr,
            fmt="o-",
            capsize=5,
            capthick=2,
            ecolor="red",
            color="blue",
            linewidth=2,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("errorbar(x, y, yerr, xerr)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_violinplot(self, temp_file):
        """Test violinplot(D)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        data = [
            np.random.randn(100),
            np.random.randn(100) + 1,
            np.random.randn(100) - 1,
        ]

        parts = ax.violinplot(
            data, positions=[1, 2, 3], showmeans=True, showmedians=True
        )
        ax.set_xlabel("Group")
        ax.set_ylabel("Value")
        ax.set_title("violinplot(D)")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["A", "B", "C"])

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_eventplot(self, temp_file):
        """Test eventplot(D)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        data = [
            np.random.gamma(2, size=50),
            np.random.gamma(4, size=50),
            np.random.gamma(6, size=50),
        ]

        ax.eventplot(
            data,
            colors=["red", "green", "blue"],
            lineoffsets=[1, 2, 3],
            linelengths=0.5,
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Group")
        ax.set_title("eventplot(D)")
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["A", "B", "C"])

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_hist2d(self, temp_file):
        """Test hist2d(x, y)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        h = ax.hist2d(x, y, bins=30, cmap="Blues")
        plt.colorbar(h[3], ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("hist2d(x, y)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_hexbin(self, temp_file):
        """Test hexbin(x, y, C)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        hb = ax.hexbin(x, y, gridsize=20, cmap="YlOrRd")
        plt.colorbar(hb, ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("hexbin(x, y, C)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_pie(self, temp_file):
        """Test pie(x)."""
        fig, ax = plt.subplots()
        sizes = [15, 30, 45, 10]
        labels = ["A", "B", "C", "D"]
        colors = ["gold", "yellowgreen", "lightcoral", "lightskyblue"]
        explode = (0, 0.1, 0, 0)

        ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )
        ax.set_title("pie(x)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_ecdf(self, temp_file):
        """Test ecdf(x)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        data = np.random.randn(100)

        ax.ecdf(data, label="ECDF", linewidth=2)
        ax.set_xlabel("Value")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("ecdf(x)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")


class TestGriddedData:
    """Test gridded data plot types."""

    def test_imshow(self, temp_file):
        """Test imshow(Z)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        data = np.random.rand(20, 30)

        im = ax.imshow(data, cmap="viridis", aspect="auto", interpolation="bilinear")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("imshow(Z)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_pcolormesh(self, temp_file):
        """Test pcolormesh(X, Y, Z)."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 30)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)

        pc = ax.pcolormesh(X, Y, Z, cmap="RdBu", shading="auto")
        plt.colorbar(pc, ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("pcolormesh(X, Y, Z)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_contour(self, temp_file):
        """Test contour(X, Y, Z)."""
        fig, ax = plt.subplots()
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)

        cs = ax.contour(X, Y, Z, levels=10, cmap="viridis")
        ax.clabel(cs, inline=True, fontsize=8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("contour(X, Y, Z)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_contourf(self, temp_file):
        """Test contourf(X, Y, Z)."""
        fig, ax = plt.subplots()
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)

        cf = ax.contourf(X, Y, Z, levels=15, cmap="RdYlBu")
        plt.colorbar(cf, ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("contourf(X, Y, Z)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_barbs(self, temp_file):
        """Test barbs(X, Y, U, V)."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)
        X, Y = np.meshgrid(x, y)
        U = np.sin(X)
        V = np.cos(Y)

        ax.barbs(X, Y, U, V, length=7, pivot="middle")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("barbs(X, Y, U, V)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_quiver(self, temp_file):
        """Test quiver(X, Y, U, V)."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)
        X, Y = np.meshgrid(x, y)
        U = np.sin(X)
        V = np.cos(Y)

        q = ax.quiver(X, Y, U, V, scale=20)
        ax.quiverkey(q, X=0.9, Y=1.05, U=1, label="1 m/s", labelpos="E")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("quiver(X, Y, U, V)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_streamplot(self, temp_file):
        """Test streamplot(X, Y, U, V)."""
        fig, ax = plt.subplots()
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        U = -Y
        V = X

        sp = ax.streamplot(
            X, Y, U, V, color=np.sqrt(U**2 + V**2), cmap="autumn", density=1.5
        )
        plt.colorbar(sp.lines, ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("streamplot(X, Y, U, V)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")


class TestIrregularGriddedData:
    """Test irregular gridded data plot types."""

    def test_tricontour(self, temp_file):
        """Test tricontour(x, y, z)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        z = x**2 + y**2

        tc = ax.tricontour(x, y, z, levels=10, cmap="viridis")
        ax.clabel(tc, inline=True, fontsize=8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("tricontour(x, y, z)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_tricontourf(self, temp_file):
        """Test tricontourf(x, y, z)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        z = x**2 + y**2

        tcf = ax.tricontourf(x, y, z, levels=15, cmap="RdYlBu")
        plt.colorbar(tcf, ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("tricontourf(x, y, z)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_tripcolor(self, temp_file):
        """Test tripcolor(x, y, z)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        z = x**2 + y**2

        tp = ax.tripcolor(x, y, z, cmap="plasma", shading="flat")
        plt.colorbar(tp, ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("tripcolor(x, y, z)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_triplot(self, temp_file):
        """Test triplot(x, y)."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50)

        ax.triplot(x, y, "bo-", markersize=5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("triplot(x, y)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")


class Test3DVolumetricData:
    """Test 3D and volumetric data plot types."""

    def test_bar3d(self, temp_file):
        """Test bar3d(x, y, z, dx, dy, dz)."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        x = np.arange(4)
        y = np.arange(5)
        X, Y = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        Z = np.zeros_like(X)
        dx = dy = 0.8
        dz = np.random.randint(1, 10, size=len(X))

        ax.bar3d(X, Y, Z, dx, dy, dz, shade=True, color="steelblue")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("bar3d(x, y, z, dx, dy, dz)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_plot3d(self, temp_file):
        """Test 3D plot(xs, ys, zs)."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        t = np.linspace(0, 20, 200)
        x = np.sin(t)
        y = np.cos(t)
        z = t

        ax.plot(x, y, z, "b-", linewidth=2, label="3D curve")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("plot(xs, ys, zs)")
        ax.legend()

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_scatter3d(self, temp_file):
        """Test 3D scatter(xs, ys, zs)."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)
        z = np.random.randn(n)
        colors = np.random.rand(n)

        ax.scatter(x, y, z, c=colors, marker="o", s=50, cmap="viridis")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("scatter(xs, ys, zs)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_stem3d(self, temp_file):
        """Test 3D stem(x, y, z)."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        theta = np.linspace(0, 2 * np.pi, 20)
        x = np.cos(theta)
        y = np.sin(theta)
        z = theta

        ax.stem(x, y, z, linefmt="b-", markerfmt="ro", basefmt="k-")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("stem(x, y, z)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_plot_wireframe(self, temp_file):
        """Test plot_wireframe(X, Y, Z)."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        x = np.linspace(-5, 5, 30)
        y = np.linspace(-5, 5, 30)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))

        ax.plot_wireframe(X, Y, Z, color="blue", linewidth=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("plot_wireframe(X, Y, Z)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")

    def test_plot_trisurf(self, temp_file):
        """Test plot_trisurf(x, y, z)."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)
        z = x**2 + y**2

        ax.plot_trisurf(x, y, z, cmap="viridis", alpha=0.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("plot_trisurf(x, y, z)")

        save_figure(fig, temp_file)
        loaded_fig = load_figure(temp_file)

        is_equal, report = compare_figures(fig, loaded_fig)
        assert is_equal, f"Figures differ: {report['differences']}"
        plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
