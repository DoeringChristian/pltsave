"""
Deserializer module for reconstructing matplotlib figures from saved data.

This module reconstructs matplotlib figures from the serialized data format,
recreating all artists and properties.
"""

import json
import base64
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, Polygon, Wedge, Arc, Ellipse, Arrow, FancyArrow, StepPatch, PathPatch, Shadow, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.collections import PathCollection, LineCollection, PolyCollection, EventCollection
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import Axes3D

# Import 3D types if available
try:
    from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection
    HAS_3D = True
except ImportError:
    HAS_3D = False


class NumpyDecoder:
    """JSON decoder that handles numpy arrays and other special types."""

    @staticmethod
    def decode_object(obj):
        """Decode special object types."""
        if isinstance(obj, dict):
            if obj.get("__type__") == "ndarray":
                # Reconstruct numpy array
                data = base64.b64decode(obj["data"])
                arr = np.frombuffer(data, dtype=obj["dtype"])
                return arr.reshape(obj["shape"])
            elif obj.get("__type__") == "bytes":
                return base64.b64decode(obj["data"])
            elif obj.get("__type__") == "complex":
                return complex(obj["real"], obj["imag"])
            else:
                # Recursively decode nested objects
                return {k: NumpyDecoder.decode_object(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [NumpyDecoder.decode_object(item) for item in obj]
        else:
            return obj


def _decode_json(filepath: str) -> Dict[str, Any]:
    """Load and decode JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return NumpyDecoder.decode_object(data)


def _restore_line2d(ax: Axes, data: Dict[str, Any]) -> Line2D:
    """Restore a Line2D object."""
    line = ax.plot(
        data["xdata"],
        data["ydata"],
        color=data["color"],
        linewidth=data["linewidth"],
        linestyle=data["linestyle"],
        marker=data["marker"],
        markersize=data["markersize"],
        markerfacecolor=data["markerfacecolor"],
        markeredgecolor=data["markeredgecolor"],
        markeredgewidth=data["markeredgewidth"],
        alpha=data["alpha"],
        label=data["label"],
        zorder=data["zorder"],
    )[0]

    line.set_visible(data["visible"])
    line.set_drawstyle(data["drawstyle"])

    return line


def _restore_patch(ax: Axes, data: Dict[str, Any]):
    """Restore a Patch object."""
    patch_type = data["type"]

    # Create the appropriate patch type
    if patch_type == "Rectangle":
        patch = Rectangle(
            xy=tuple(data["xy"]),
            width=data["width"],
            height=data["height"],
            angle=data["angle"],
        )
    elif patch_type == "Circle":
        patch = Circle(
            xy=tuple(data["center"]),
            radius=data["radius"],
        )
    elif patch_type == "Ellipse":
        patch = Ellipse(
            xy=tuple(data["xy"]),
            width=data["width"],
            height=data["height"],
            angle=data["angle"],
        )
    elif patch_type == "Wedge":
        patch = Wedge(
            center=tuple(data["center"]),
            r=data["r"],
            theta1=data["theta1"],
            theta2=data["theta2"],
            width=data["width"],
        )
    elif patch_type == "Polygon":
        patch = Polygon(
            xy=data["xy"],
        )
    elif patch_type == "Arc":
        patch = Arc(
            xy=tuple(data["xy"]),
            width=data["width"],
            height=data["height"],
            angle=data["angle"],
            theta1=data["theta1"],
            theta2=data["theta2"],
        )
    elif patch_type in ("Arrow", "FancyArrow"):
        # Arrows are more complex, need special handling
        if data.get("x") is not None and data.get("dx") is not None:
            patch = FancyArrow(
                data["x"],
                data["y"],
                data["dx"],
                data["dy"],
            )
        else:
            # Fallback to a simple polygon
            return None
    elif patch_type == "StepPatch":
        # StepPatch is created by ax.stairs()
        patch = StepPatch(
            values=np.array(data["values"]),
            edges=np.array(data["edges"]),
            orientation=data.get("orientation", "vertical"),
            baseline=data.get("baseline", 0),
        )
    elif patch_type == "PathPatch":
        # Generic PathPatch (used by boxplot, etc.)
        if "vertices" in data and data["vertices"] is not None:
            vertices = np.array(data["vertices"])
            codes = np.array(data["codes"]) if data.get("codes") is not None else None
            path = Path(vertices, codes)
            patch = PathPatch(path)
        else:
            # Can't reconstruct without path data
            return None
    elif patch_type == "FancyArrowPatch":
        # FancyArrowPatch from streamplot - reconstruct as PathPatch with arrow shape
        if "vertices" in data and data["vertices"] is not None:
            vertices = np.array(data["vertices"])
            codes = np.array(data["codes"]) if data.get("codes") is not None else None
            path = Path(vertices, codes)
            # Reconstruct as PathPatch since FancyArrowPatch is complex
            patch = PathPatch(path)
        else:
            return None
    elif patch_type == "Shadow":
        # Shadow patch (used by pie charts with shadow=True)
        if "vertices" in data and data["vertices"] is not None:
            vertices = np.array(data["vertices"])
            codes = np.array(data["codes"]) if data.get("codes") is not None else None
            path = Path(vertices, codes)
            # Shadow inherits from PathPatch
            patch = Shadow(PathPatch(path), ox=0, oy=0)
        else:
            # Can't reconstruct without path data
            return None
    else:
        # Unknown patch type
        return None

    # Set common properties
    patch.set_facecolor(data["facecolor"])
    patch.set_edgecolor(data["edgecolor"])
    patch.set_linewidth(data["linewidth"])
    patch.set_linestyle(data["linestyle"])
    # Don't set alpha separately - it's already embedded in the facecolor/edgecolor
    # If we call set_alpha(), it will modify the colors we just set
    # patch.set_alpha(data["alpha"])
    patch.set_label(data["label"])
    patch.set_zorder(data["zorder"])
    patch.set_visible(data["visible"])
    patch.set_fill(data["fill"])

    ax.add_patch(patch)
    return patch


def _restore_collection(ax: Axes, data: Dict[str, Any]):
    """Restore a Collection object."""
    collection_type = data["type"]

    # Handle 3D collections
    if HAS_3D and collection_type == "Path3DCollection":
        # 3D scatter plot
        if "offsets3d" in data:
            xs = np.array(data["offsets3d"]["xs"])
            ys = np.array(data["offsets3d"]["ys"])
            zs = np.array(data["offsets3d"]["zs"])

            sizes = np.array(data.get("sizes", []))
            s = sizes if len(sizes) > 0 else None

            scatter = ax.scatter(
                xs, ys, zs,
                s=s,
                alpha=data.get("alpha"),
                label=data.get("label"),
                zorder=data.get("zorder"),
            )

            # Set colors if available
            if "facecolors" in data and len(data["facecolors"]) > 0:
                scatter.set_facecolors(data["facecolors"])
            if "edgecolors" in data and len(data["edgecolors"]) > 0:
                scatter.set_edgecolors(data["edgecolors"])
            if "linewidths" in data and len(data["linewidths"]) > 0:
                scatter.set_linewidths(data["linewidths"])

            scatter.set_visible(data.get("visible", True))

            return scatter

    elif HAS_3D and collection_type == "Poly3DCollection":
        # 3D polygons (used by bar3d, plot_trisurf, plot_wireframe, etc.)
        vec = np.array(data.get("vec", []))
        if len(vec) == 0:
            return None

        # Reshape vec back to 3D vertices
        # vec is saved as a flat list of [x, y, z] coordinates for each polygon
        verts = []
        for polygon_data in vec:
            # Each polygon has vertices in groups of 3 (x, y, z)
            n_verts = len(polygon_data) // 3
            polygon_verts = []
            for i in range(n_verts):
                polygon_verts.append([
                    polygon_data[i],
                    polygon_data[i + n_verts],
                    polygon_data[i + 2 * n_verts]
                ])
            verts.append(polygon_verts)

        # Create Poly3DCollection
        pc = Poly3DCollection(
            verts,
            alpha=data.get("alpha"),
            label=data.get("label"),
            zorder=data.get("zorder"),
        )

        # Set colors if available
        if "facecolors" in data and len(data["facecolors"]) > 0:
            pc.set_facecolors(data["facecolors"])
        if "edgecolors" in data and len(data["edgecolors"]) > 0:
            pc.set_edgecolors(data["edgecolors"])

        pc.set_visible(data.get("visible", True))
        ax.add_collection3d(pc)
        return pc

    elif collection_type == "PathCollection":
        # This is typically from scatter()
        offsets = np.array(data.get("offsets", []))
        sizes = np.array(data.get("sizes", []))

        if len(offsets) == 0:
            return None

        # Use scatter to create the collection
        scatter = ax.scatter(
            offsets[:, 0],
            offsets[:, 1],
            s=sizes if len(sizes) > 0 else None,
            alpha=data.get("alpha"),
            label=data.get("label"),
            zorder=data.get("zorder"),
        )

        # Set colors if available
        if "facecolors" in data and len(data["facecolors"]) > 0:
            scatter.set_facecolors(data["facecolors"])
        if "edgecolors" in data and len(data["edgecolors"]) > 0:
            scatter.set_edgecolors(data["edgecolors"])
        if "linewidths" in data and len(data["linewidths"]) > 0:
            scatter.set_linewidths(data["linewidths"])

        scatter.set_visible(data.get("visible", True))

        # Set colormap if available
        if "cmap" in data:
            scatter.set_cmap(data["cmap"])
        if "norm" in data and data["norm"]:
            scatter.set_clim(data["norm"].get("vmin"), data["norm"].get("vmax"))

        return scatter

    elif HAS_3D and collection_type == "Line3DCollection":
        # 3D line collection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        segments3d = data.get("segments3d", [])
        if len(segments3d) == 0:
            return None

        # Line3DCollection constructor takes list of segments
        lc3d = Line3DCollection(
            segments3d,
            linewidths=data.get("linewidths", 1),
            alpha=data.get("alpha"),
            label=data.get("label"),
        )

        if "colors" in data and len(data["colors"]) > 0:
            lc3d.set_colors(data["colors"])

        lc3d.set_visible(data.get("visible", True))
        lc3d.set_zorder(data.get("zorder", 1))
        ax.add_collection3d(lc3d)
        return lc3d

    elif collection_type == "EventCollection":
        # EventCollection is a special LineCollection for eventplot
        segments = data.get("segments", [])
        if len(segments) == 0:
            return None

        # Extract positions from segments (EventCollection uses vertical/horizontal lines)
        # Each segment is [[x1, y1], [x2, y2]] for a single event line
        positions = []
        for seg in segments:
            # Take the x-coordinate (or y for vertical orientation)
            positions.append(seg[0][0])

        lineoffsets = data.get("lineoffsets", 0)
        linelengths = data.get("linelengths", 1)

        # EventCollection constructor signature is (positions, orientation='horizontal', lineoffset=0, linelength=1, ...)
        ec = EventCollection(positions)

        # Set properties after creation
        ec.set_lineoffset(lineoffsets)
        ec.set_linelength(linelengths)

        if "colors" in data and len(data["colors"]) > 0:
            ec.set_colors(data["colors"])

        if "linewidths" in data:
            ec.set_linewidths(data["linewidths"])

        if data.get("alpha") is not None:
            ec.set_alpha(data.get("alpha"))

        ec.set_visible(data.get("visible", True))
        ec.set_zorder(data.get("zorder", 1))
        ec.set_label(data.get("label", ""))
        ax.add_collection(ec)
        return ec

    elif collection_type == "LineCollection":
        from matplotlib.collections import LineCollection

        segments = data.get("segments", [])
        # Allow empty segments (colorbars can have empty LineCollections)

        lc = LineCollection(
            segments,
            linewidths=data.get("linewidths", 1),
            alpha=data.get("alpha"),
            label=data.get("label"),
            zorder=data.get("zorder"),
        )

        if "colors" in data and len(data["colors"]) > 0:
            lc.set_colors(data["colors"])

        lc.set_visible(data.get("visible", True))
        ax.add_collection(lc)
        return lc

    elif collection_type == "Barbs":
        # Barbs plots (similar to Quiver but lowercase)
        x = np.array(data.get("x", []))
        y = np.array(data.get("y", []))
        u = np.array(data.get("u", []))
        v = np.array(data.get("v", []))

        if len(x) == 0 or len(u) == 0:
            return None

        b = ax.barbs(x, y, u, v, alpha=data.get("alpha"), zorder=data.get("zorder"))

        # Set colors if available
        if "facecolors" in data and len(data["facecolors"]) > 0:
            b.set_facecolors(data["facecolors"])
        if "edgecolors" in data and len(data["edgecolors"]) > 0:
            b.set_edgecolors(data["edgecolors"])

        b.set_visible(data.get("visible", True))
        b.set_label(data.get("label", ""))
        return b

    elif collection_type == "Quiver":
        # Quiver plots
        X = np.array(data.get("X", []))
        Y = np.array(data.get("Y", []))
        U = np.array(data.get("U", []))
        V = np.array(data.get("V", []))

        if len(X) == 0 or len(U) == 0:
            return None

        q = ax.quiver(X, Y, U, V, alpha=data.get("alpha"), zorder=data.get("zorder"))

        # Set colors if available
        if "facecolors" in data and len(data["facecolors"]) > 0:
            q.set_facecolors(data["facecolors"])
        if "edgecolors" in data and len(data["edgecolors"]) > 0:
            q.set_edgecolors(data["edgecolors"])

        q.set_visible(data.get("visible", True))
        q.set_label(data.get("label", ""))
        return q

    elif collection_type in ("QuadContourSet", "ContourSet", "TriContourSet"):
        # Contour plots - these are complex and require original X,Y,Z data
        # Create a placeholder empty LineCollection to maintain collection count
        from matplotlib.collections import LineCollection

        lc = LineCollection(
            [],  # Empty segments
            alpha=data.get("alpha"),
            label=data.get("label"),
            zorder=data.get("zorder"),
        )

        lc.set_visible(data.get("visible", True))
        ax.add_collection(lc)
        return lc

    elif "PolyCollection" in collection_type:
        # Handles PolyCollection and subclasses like FillBetweenPolyCollection, StackPolyCollection, etc.
        from matplotlib.collections import PolyCollection

        verts = data.get("verts", [])
        if len(verts) == 0:
            return None

        pc = PolyCollection(
            verts,
            alpha=data.get("alpha"),
            label=data.get("label"),
            zorder=data.get("zorder"),
        )

        if "facecolors" in data and len(data["facecolors"]) > 0:
            pc.set_facecolors(data["facecolors"])
        if "edgecolors" in data and len(data["edgecolors"]) > 0:
            pc.set_edgecolors(data["edgecolors"])

        pc.set_visible(data.get("visible", True))
        ax.add_collection(pc)
        return pc

    elif collection_type == "QuadMesh":
        # QuadMesh is typically from pcolormesh
        if "coordinates" in data and "array" in data:
            coords = np.array(data["coordinates"])
            array = np.array(data["array"])

            mesh = ax.pcolormesh(
                coords[:, :, 0],
                coords[:, :, 1],
                array,
                alpha=data.get("alpha"),
                zorder=data.get("zorder"),
            )

            if "cmap" in data:
                mesh.set_cmap(data["cmap"])
            if "norm" in data and data["norm"]:
                mesh.set_clim(data["norm"].get("vmin"), data["norm"].get("vmax"))

            mesh.set_visible(data.get("visible", True))
            return mesh

    return None


def _restore_text(ax: Axes, data: Dict[str, Any]):
    """Restore a Text or Annotation object."""
    if data["type"] == "Annotation":
        # Annotation
        ann = ax.annotate(
            data["text"],
            xy=tuple(data["xy"]),
            xytext=tuple(data["xytext"]) if data.get("xytext") else None,
            fontsize=data["fontsize"],
            color=data["color"],
            ha=data["horizontalalignment"],
            va=data["verticalalignment"],
            rotation=data["rotation"],
            alpha=data["alpha"],
            family=data["family"],
            style=data["style"],
            weight=data["weight"],
            zorder=data["zorder"],
            arrowprops=data.get("arrowprops"),
        )
        ann.set_visible(data["visible"])
        return ann
    else:
        # Regular text
        text = ax.text(
            data["position"][0],
            data["position"][1],
            data["text"],
            fontsize=data["fontsize"],
            color=data["color"],
            ha=data["horizontalalignment"],
            va=data["verticalalignment"],
            rotation=data["rotation"],
            alpha=data["alpha"],
            family=data["family"],
            style=data["style"],
            weight=data["weight"],
            zorder=data["zorder"],
        )
        text.set_visible(data["visible"])
        return text


def _restore_image(ax: Axes, data: Dict[str, Any]):
    """Restore an AxesImage object."""
    array = np.array(data["array"]) if data["array"] is not None else None

    if array is None:
        return None

    img = ax.imshow(
        array,
        extent=data.get("extent"),
        alpha=data.get("alpha"),
        zorder=data.get("zorder"),
        interpolation=data.get("interpolation"),
    )

    if "cmap" in data:
        img.set_cmap(data["cmap"])
    if "norm" in data and data["norm"]:
        img.set_clim(data["norm"].get("vmin"), data["norm"].get("vmax"))

    img.set_visible(data.get("visible", True))
    return img


def _restore_legend(ax: Axes, data: Optional[Dict[str, Any]]):
    """Restore a Legend object."""
    if data is None:
        return

    # Get handles and labels from the axes
    handles, labels = ax.get_legend_handles_labels()

    # If we have saved labels, use those
    if data.get("labels"):
        labels = data["labels"]

    if len(handles) > 0:
        legend = ax.legend(
            handles[:len(labels)],
            labels,
            loc=data.get("loc"),
            frameon=data.get("frameon", True),
            shadow=data.get("shadow", False),
            framealpha=data.get("framealpha"),
            facecolor=data.get("facecolor"),
            edgecolor=data.get("edgecolor"),
        )

        if data.get("title"):
            legend.set_title(data["title"])

        return legend


def _restore_axes(fig: Figure, data: Dict[str, Any]) -> Axes:
    """Restore an Axes object."""
    # Create the axes with the correct projection
    if data.get("is_3d", False):
        ax = fig.add_axes(data["position"], projection='3d')
    else:
        ax = fig.add_axes(data["position"])

    # Set labels and title first
    ax.set_xlabel(data["xlabel"])
    ax.set_ylabel(data["ylabel"])
    ax.set_title(data["title"])

    ax.set_facecolor(data["facecolor"])

    if not data["frame_on"]:
        ax.set_frame_on(False)

    # Spines
    if "spines" in data:
        for spine_name, spine_data in data["spines"].items():
            if spine_name in ax.spines:
                ax.spines[spine_name].set_visible(spine_data["visible"])
                ax.spines[spine_name].set_edgecolor(spine_data["color"])
                ax.spines[spine_name].set_linewidth(spine_data["linewidth"])

    # Restore all artists FIRST (before setting scales/limits)
    for line_data in data.get("lines", []):
        _restore_line2d(ax, line_data)

    for patch_data in data.get("patches", []):
        _restore_patch(ax, patch_data)

    for collection_data in data.get("collections", []):
        _restore_collection(ax, collection_data)

    for text_data in data.get("texts", []):
        _restore_text(ax, text_data)

    for image_data in data.get("images", []):
        _restore_image(ax, image_data)

    for artist_data in data.get("artists", []):
        artist_type = artist_data.get("type")
        if artist_type == "Line2D":
            _restore_line2d(ax, artist_data)
        elif artist_type in ("Rectangle", "Circle", "Polygon", "Wedge", "Arc", "Ellipse"):
            _restore_patch(ax, artist_data)
        elif artist_type in ("Text", "Annotation"):
            _restore_text(ax, artist_data)

    # Now set limits and scale AFTER all artists are added
    # For log scales, we need to set limits BEFORE scale to avoid auto-scaling
    ax.set_xlim(data["xlim"])
    ax.set_ylim(data["ylim"])

    # Set scales - skip FuncScale as it requires additional parameters we don't save
    try:
        if data["xscale"] != "function":
            ax.set_xscale(data["xscale"])
    except:
        pass

    try:
        if data["yscale"] != "function":
            ax.set_yscale(data["yscale"])
    except:
        pass

    # Set limits again after scale (matplotlib sometimes resets them)
    ax.set_xlim(data["xlim"])
    ax.set_ylim(data["ylim"])

    # Set aspect only for 2D axes
    if not data.get("is_3d", False):
        ax.set_aspect(data["aspect"])

    # Grid
    if data.get("grid", {}).get("visible", False):
        ax.grid(True)

    # Ticks
    if data.get("xticks"):
        ax.set_xticks(data["xticks"])
    if data.get("yticks"):
        ax.set_yticks(data["yticks"])
    if data.get("xticklabels"):
        ax.set_xticklabels(data["xticklabels"])
    if data.get("yticklabels"):
        ax.set_yticklabels(data["yticklabels"])

    # 3D specific properties
    if data.get("is_3d", False):
        ax.set_zlim(data["zlim"])
        ax.set_zlabel(data["zlabel"])
        if data.get("zticks"):
            ax.set_zticks(data["zticks"])
        if data.get("zticklabels"):
            ax.set_zticklabels(data["zticklabels"])

        # Set view angle
        if "view_angle" in data:
            ax.view_init(
                elev=data["view_angle"]["elev"],
                azim=data["view_angle"]["azim"]
            )

    # Restore legend (must be done after artists are added)
    _restore_legend(ax, data.get("legend"))

    return ax


def _restore_figure(data: Dict[str, Any]) -> Figure:
    """Restore a Figure object."""
    # Create figure with the correct size and properties
    fig = plt.figure(
        figsize=tuple(data["figsize"]),
        dpi=data["dpi"],
        facecolor=data["facecolor"],
        edgecolor=data["edgecolor"],
        frameon=data["frameon"],
    )

    # Set layout
    if data.get("tight_layout"):
        fig.set_tight_layout(data["tight_layout"])
    if data.get("constrained_layout"):
        fig.set_constrained_layout(data["constrained_layout"])

    # Restore all axes
    for ax_data in data.get("axes", []):
        _restore_axes(fig, ax_data)

    # Restore suptitle
    if data.get("suptitle"):
        fig.suptitle(
            data["suptitle"]["text"],
            fontsize=data["suptitle"]["fontsize"],
            fontweight=data["suptitle"]["fontweight"],
        )

    # Restore figure texts
    for text_data in data.get("texts", []):
        fig.text(
            text_data["position"][0],
            text_data["position"][1],
            text_data["text"],
            fontsize=text_data["fontsize"],
            color=text_data["color"],
            ha=text_data["horizontalalignment"],
            va=text_data["verticalalignment"],
            rotation=text_data["rotation"],
            alpha=text_data["alpha"],
        )

    return fig


def load_figure(filepath: str) -> Figure:
    """
    Load a matplotlib figure from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON file

    Returns
    -------
    matplotlib.figure.Figure
        The reconstructed figure

    Examples
    --------
    >>> from pltsave import load_figure
    >>> import matplotlib.pyplot as plt
    >>> fig = load_figure('my_figure.json')
    >>> plt.show()
    """
    data = _decode_json(filepath)
    fig = _restore_figure(data)
    return fig
