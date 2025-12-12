"""Viewer script to load and display a saved figure interactively."""

import sys
import matplotlib

# matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from pltsave import load_figure


def main():
    """Load and display a saved figure."""
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "example_figure.json"

    print(f"Loading figure from '{filename}'...")
    fig = load_figure(filename)
    print("Figure loaded successfully!")
    plt.show()


if __name__ == "__main__":
    main()
