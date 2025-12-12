# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pltsave is a matplotlib figure serialization library that saves/loads figures by serializing their internal data structure to JSON, not just as images. This enables interactive reloading and manipulation of previously saved figures.

## Development Commands

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pltsave.py -v

# Run specific test class
pytest tests/test_pltsave.py::TestBasicPlots -v

# Run specific test
pytest tests/test_pltsave.py::TestBasicPlots::test_simple_line_plot -v

# Run with coverage
pytest tests/ --cov=pltsave --cov-report=html

# Run demo
python main.py

# View a saved figure interactively
python viewer.py                    # loads example_figure.json by default
python viewer.py my_figure.json     # load specific file
```

## Architecture

The library is organized into three core modules in `pltsave/`:

- **serializer.py**: Extracts data from matplotlib figures via introspection. Walks the figure hierarchy (Figure -> Axes -> Artists) and serializes each component. Uses `NumpyEncoder` for JSON encoding of numpy arrays (base64-encoded) and special types.

- **deserializer.py**: Reconstructs figures from serialized JSON. Uses `NumpyDecoder` for decoding, then rebuilds figures by creating matplotlib objects with saved properties. Artists are restored before axis limits/scales are set.

- **comparator.py**: Compares two figures for equivalence with configurable tolerance. Used for testing save/load round-trips. Handles edge cases like log scale limits and PolyCollection subclass variations.

## Key Design Patterns

- Each matplotlib artist type has paired `_serialize_*` and `_restore_*` functions
- 3D support is conditional (`HAS_3D` flag) based on mpl_toolkits.mplot3d availability
- Colors are serialized to lists; numpy arrays are base64-encoded with dtype/shape metadata
- Comparison uses lenient tolerances for axis limits due to autoscaling behavior

## Public API

```python
from pltsave import save_figure, load_figure, compare_figures

save_figure(fig, 'output.json')  # Serialize figure to JSON
fig = load_figure('output.json')  # Reconstruct figure from JSON
is_equal, report = compare_figures(fig1, fig2)  # Compare two figures
```
