# pltsave

A comprehensive matplotlib figure serialization library that allows you to save and load matplotlib figures by serializing their internal data structure, not just as images.

## Features

- **Full Figure Serialization**: Saves all matplotlib figure data, not just images
- **Interactive Reloading**: Load figures back for interactive manipulation
- **Comprehensive Support**: Handles all major matplotlib features:
  - Line plots, scatter plots, bar charts, histograms
  - 2D and 3D plots
  - Images and heatmaps
  - Patches (rectangles, circles, polygons, wedges, etc.)
  - Text and annotations with arrows
  - Legends and colorbars
  - Multiple subplots
  - Custom axes properties (scales, ticks, labels, etc.)
  - And much more!
- **Figure Comparison**: Built-in utilities to verify saved/loaded figures match originals
- **JSON Format**: Human-readable serialization format

## Installation

### From GitHub

Install directly from the GitHub repository:

```bash
pip install git+https://github.com/DoeringChristian/pltsave.git
```

### For Development

Clone the repository and install in editable mode:

```bash
git clone https://github.com/DoeringChristian/pltsave.git
cd pltsave
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from pltsave import save_figure, load_figure, compare_figures

# Create a figure
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Trigonometric Functions')
ax.legend()

# Save the figure
save_figure(fig, 'my_figure.json')

# Load it back
loaded_fig = load_figure('my_figure.json')

# Compare to verify they match
is_equal, report = compare_figures(fig, loaded_fig)
print(f"Figures match: {is_equal}")

# Show the loaded figure
plt.show()
```

## Usage

### Saving Figures

```python
from pltsave import save_figure

# Save any matplotlib figure
save_figure(fig, 'output.json')
```

The figure is serialized to JSON format with all data preserved:
- Figure properties (size, DPI, colors)
- All axes and their properties
- All artists (lines, patches, collections, text, images)
- Legends, colorbars, annotations
- 3D plot data and viewing angles
- Custom axis properties (scales, limits, ticks)

### Loading Figures

```python
from pltsave import load_figure

# Load a saved figure
fig = load_figure('output.json')

# The figure is fully interactive
plt.show()
```

### Comparing Figures

```python
from pltsave import compare_figures

# Compare two figures
is_equal, report = compare_figures(original_fig, loaded_fig)

if is_equal:
    print("Figures are identical!")
else:
    print("Differences found:")
    for diff in report['differences']:
        print(f"  - {diff}")
```

## Examples

### Example 1: Complex Multi-Panel Figure

```python
import numpy as np
import matplotlib.pyplot as plt
from pltsave import save_figure, load_figure

fig = plt.figure(figsize=(12, 8))

# Line plot
ax1 = plt.subplot(2, 2, 1)
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), 'r-', linewidth=2, label='sin')
ax1.plot(x, np.cos(x), 'b--', linewidth=2, label='cos')
ax1.legend()
ax1.set_title('Trig Functions')
ax1.grid(True)

# Scatter plot
ax2 = plt.subplot(2, 2, 2)
n = 100
x_scatter = np.random.randn(n)
y_scatter = np.random.randn(n)
colors = np.random.rand(n)
ax2.scatter(x_scatter, y_scatter, c=colors, s=50, cmap='viridis')
ax2.set_title('Scatter Plot')

# Bar chart
ax3 = plt.subplot(2, 2, 3)
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
ax3.bar(categories, values, color='coral')
ax3.set_title('Bar Chart')

# Heatmap
ax4 = plt.subplot(2, 2, 4)
data = np.random.rand(20, 20)
im = ax4.imshow(data, cmap='plasma')
ax4.set_title('Heatmap')
plt.colorbar(im, ax=ax4)

fig.suptitle('Multi-Panel Figure', fontsize=16)
fig.tight_layout()

# Save and reload
save_figure(fig, 'multi_panel.json')
loaded_fig = load_figure('multi_panel.json')
plt.show()
```

### Example 2: 3D Plot

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pltsave import save_figure, load_figure

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create 3D surface
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface')

# Save and reload - 3D view angle is preserved!
save_figure(fig, '3d_surface.json')
loaded_fig = load_figure('3d_surface.json')
plt.show()
```

### Example 3: Annotations and Shapes

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from pltsave import save_figure, load_figure

fig, ax = plt.subplots(figsize=(8, 8))

# Add shapes
circle = Circle((0.3, 0.5), 0.15, facecolor='red', edgecolor='black', linewidth=2)
rect = Rectangle((0.5, 0.3), 0.2, 0.4, facecolor='blue', alpha=0.5)
triangle = Polygon([[0.1, 0.1], [0.3, 0.3], [0.1, 0.3]], facecolor='green')

ax.add_patch(circle)
ax.add_patch(rect)
ax.add_patch(triangle)

# Add annotations
ax.annotate('Circle', xy=(0.3, 0.5), xytext=(0.15, 0.7),
           arrowprops=dict(arrowstyle='->', color='red', lw=2),
           fontsize=12)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_title('Shapes and Annotations')

# Save and reload
save_figure(fig, 'shapes.json')
loaded_fig = load_figure('shapes.json')
plt.show()
```

## Running Tests

The project includes a comprehensive test suite covering all matplotlib features:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pltsave --cov-report=html

# Run specific test file
pytest tests/test_pltsave.py -v

# Run specific test class
pytest tests/test_pltsave.py::TestBasicPlots -v

# Run specific test
pytest tests/test_pltsave.py::TestBasicPlots::test_simple_line_plot -v
```

## Demo

Run the demo script to see the library in action:

```bash
python main.py
```

This will:
1. Create a comprehensive figure with multiple plot types
2. Save it to `demo_figure.json`
3. Load it back
4. Compare original and loaded figures
5. Display both figures interactively

## How It Works

### Serialization Process

1. **Figure Introspection**: The library walks through the figure's object hierarchy
2. **Data Extraction**: Extracts all relevant data from each component:
   - Figure: size, DPI, colors, layout
   - Axes: limits, scales, labels, ticks, spines
   - Artists: line data, patch geometry, collection points, text content
   - Properties: colors, line styles, markers, fonts, etc.
3. **JSON Encoding**: Converts data to JSON using custom encoders for numpy arrays and special types

### Deserialization Process

1. **JSON Decoding**: Loads JSON and reconstructs numpy arrays
2. **Figure Recreation**: Creates new figure with saved properties
3. **Artist Reconstruction**: Rebuilds all artists with their original data and properties
4. **Property Restoration**: Applies all saved properties to recreate the original appearance

## Supported Features

### Plot Types
-  Line plots (plot, semilogx, semilogy, loglog)
-  Scatter plots
-  Bar charts (bar, barh)
-  Histograms
-  Images (imshow)
-  Contour plots
-  Quiver plots
-  3D plots (line, scatter, surface, wireframe)

### Artists
-  Line2D (lines, curves)
-  Patches (Rectangle, Circle, Ellipse, Polygon, Wedge, Arc)
-  Collections (PathCollection, LineCollection, PolyCollection, QuadMesh)
-  Text and Annotations
-  Images (AxesImage)
-  Legends
-  Colorbars (partial support)

### Properties
-  Colors (named, RGB, RGBA)
-  Line styles and widths
-  Markers (type, size, colors)
-  Fonts (family, size, style, weight)
-  Axis properties (limits, scales, ticks, labels)
-  Grid settings
-  Spines
-  Transparency (alpha)
-  Z-order
-  Visibility

## Limitations

While pltsave supports a comprehensive set of matplotlib features, some advanced or less common features may not be fully supported:

- Complex custom transformations
- Some specialized plot types (polar plots, ternary plots, etc.)
- Custom renderers
- Some advanced colorbar configurations
- Interactive widgets

## Contributing

Contributions are welcome! If you find a matplotlib feature that isn't properly serialized, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

Built with:
- matplotlib - Comprehensive 2D/3D plotting
- numpy - Numerical computing
- pytest - Testing framework
