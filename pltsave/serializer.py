"""
Serializer module for extracting data from matplotlib figures.

This module provides comprehensive introspection of matplotlib figure objects,
extracting all relevant data for later reconstruction.
"""

import json
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Circle, Polygon, Wedge, FancyBboxPatch, Arc, Ellipse, Arrow, FancyArrow, StepPatch
from matplotlib.collections import PathCollection, LineCollection, PolyCollection, QuadMesh, EventCollection
from matplotlib.text import Text, Annotation
from matplotlib.image import AxesImage
from matplotlib.contour import ContourSet, QuadContourSet
from matplotlib.quiver import Quiver, Barbs
from matplotlib.legend import Legend
from matplotlib.colorbar import Colorbar
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt

# Import 3D types if available
try:
    from mpl_toolkits.mplot3d.art3d import Path3DCollection, Line3DCollection, Poly3DCollection
    HAS_3D = True
except ImportError:
    HAS_3D = False


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and other special types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": base64.b64encode(obj.tobytes()).decode('utf-8')
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, bytes):
            return {
                "__type__": "bytes",
                "data": base64.b64encode(obj).decode('utf-8')
            }
        elif isinstance(obj, complex):
            return {
                "__type__": "complex",
                "real": obj.real,
                "imag": obj.imag
            }
        elif obj is None or isinstance(obj, (int, float, str)):
            return obj
        else:
            # For other objects, try to convert to string
            return str(obj)


def _serialize_color(color) -> Optional[Any]:
    """Serialize a color value."""
    if color is None:
        return None
    if isinstance(color, str):
        return color
    if isinstance(color, (list, tuple, np.ndarray)):
        return list(color)
    return str(color)


def _serialize_transform(transform) -> Dict[str, Any]:
    """Serialize a transformation matrix."""
    if transform is None:
        return None
    try:
        matrix = transform.get_matrix()
        return {
            "matrix": matrix.tolist() if hasattr(matrix, 'tolist') else list(matrix)
        }
    except:
        return None


def _serialize_bbox(bbox: Optional[Bbox]) -> Optional[Dict[str, Any]]:
    """Serialize a bounding box."""
    if bbox is None:
        return None
    try:
        return {
            "x0": float(bbox.x0),
            "y0": float(bbox.y0),
            "x1": float(bbox.x1),
            "y1": float(bbox.y1)
        }
    except:
        return None


def _serialize_line2d(line: Line2D) -> Dict[str, Any]:
    """Serialize a Line2D object."""
    return {
        "type": "Line2D",
        "xdata": line.get_xdata().tolist(),
        "ydata": line.get_ydata().tolist(),
        "color": _serialize_color(line.get_color()),
        "linewidth": float(line.get_linewidth()),
        "linestyle": line.get_linestyle(),
        "marker": line.get_marker(),
        "markersize": float(line.get_markersize()),
        "markerfacecolor": _serialize_color(line.get_markerfacecolor()),
        "markeredgecolor": _serialize_color(line.get_markeredgecolor()),
        "markeredgewidth": float(line.get_markeredgewidth()),
        "alpha": line.get_alpha(),
        "label": line.get_label(),
        "zorder": float(line.get_zorder()),
        "visible": line.get_visible(),
        "drawstyle": line.get_drawstyle(),
    }


def _serialize_patch(patch: Patch) -> Dict[str, Any]:
    """Serialize a Patch object (Rectangle, Circle, Polygon, etc.)."""
    data = {
        "type": type(patch).__name__,
        "facecolor": _serialize_color(patch.get_facecolor()),
        "edgecolor": _serialize_color(patch.get_edgecolor()),
        "linewidth": float(patch.get_linewidth()),
        "linestyle": patch.get_linestyle(),
        "alpha": patch.get_alpha(),
        "label": patch.get_label(),
        "zorder": float(patch.get_zorder()),
        "visible": patch.get_visible(),
        "fill": patch.get_fill(),
    }

    # Add type-specific data
    if isinstance(patch, Rectangle):
        data.update({
            "xy": list(patch.get_xy()),
            "width": float(patch.get_width()),
            "height": float(patch.get_height()),
            "angle": float(patch.angle),
        })
    elif isinstance(patch, Circle):
        data.update({
            "center": list(patch.center),
            "radius": float(patch.radius),
        })
    elif isinstance(patch, Ellipse):
        data.update({
            "xy": list(patch.center),
            "width": float(patch.width),
            "height": float(patch.height),
            "angle": float(patch.angle),
        })
    elif isinstance(patch, Wedge):
        data.update({
            "center": list(patch.center),
            "r": float(patch.r),
            "theta1": float(patch.theta1),
            "theta2": float(patch.theta2),
            "width": float(patch.width) if patch.width is not None else None,
        })
    elif isinstance(patch, Polygon):
        data.update({
            "xy": patch.get_xy().tolist(),
        })
    elif isinstance(patch, (Arrow, FancyArrow)):
        data.update({
            "x": float(patch.get_x()) if hasattr(patch, 'get_x') else None,
            "y": float(patch.get_y()) if hasattr(patch, 'get_y') else None,
            "dx": getattr(patch, 'dx', None),
            "dy": getattr(patch, 'dy', None),
        })
    elif isinstance(patch, Arc):
        data.update({
            "xy": list(patch.center),
            "width": float(patch.width),
            "height": float(patch.height),
            "angle": float(patch.angle),
            "theta1": float(patch.theta1),
            "theta2": float(patch.theta2),
        })
    elif isinstance(patch, StepPatch):
        # StepPatch is a PathPatch used by ax.stairs()
        data.update({
            "values": patch._values.tolist() if hasattr(patch._values, 'tolist') else list(patch._values),
            "edges": patch._edges.tolist() if hasattr(patch._edges, 'tolist') else list(patch._edges),
            "baseline": float(patch._baseline),
            "orientation": patch.orientation,
        })
    else:
        # For any other patch type, try to get the path if available (e.g., PathPatch)
        if hasattr(patch, 'get_path'):
            try:
                path = patch.get_path()
                data.update({
                    "vertices": path.vertices.tolist(),
                    "codes": path.codes.tolist() if path.codes is not None else None,
                })
            except:
                pass

    return data


def _serialize_collection(collection) -> Dict[str, Any]:
    """Serialize a Collection object (scatter, contour, etc.)."""
    data = {
        "type": type(collection).__name__,
        "alpha": collection.get_alpha(),
        "label": collection.get_label(),
        "zorder": float(collection.get_zorder()),
        "visible": collection.get_visible(),
    }

    # Handle 3D collections first (they inherit from 2D collections)
    if HAS_3D and isinstance(collection, Path3DCollection):
        # 3D scatter plot
        if hasattr(collection, '_offsets3d'):
            xs, ys, zs = collection._offsets3d
            data["offsets3d"] = {
                "xs": xs.data.tolist() if hasattr(xs, 'data') else xs.tolist(),
                "ys": ys.data.tolist() if hasattr(ys, 'data') else ys.tolist(),
                "zs": zs.tolist() if hasattr(zs, 'tolist') else list(zs),
            }
        data["sizes"] = collection.get_sizes().tolist() if len(collection.get_sizes()) > 0 else []

        # Get colors
        try:
            facecolors = collection.get_facecolors()
            if len(facecolors) > 0:
                data["facecolors"] = facecolors.tolist()
        except:
            pass

        try:
            edgecolors = collection.get_edgecolors()
            if len(edgecolors) > 0:
                data["edgecolors"] = edgecolors.tolist()
        except:
            pass

        data["linewidths"] = collection.get_linewidths().tolist() if len(collection.get_linewidths()) > 0 else []

    elif HAS_3D and isinstance(collection, Poly3DCollection):
        # 3D surface plot
        if hasattr(collection, '_vec'):
            data["vec"] = collection._vec.tolist()
        if hasattr(collection, '_A'):
            arr = collection._A
            if arr is not None:
                data["array"] = arr.tolist()

        # Get colors
        try:
            facecolors = collection.get_facecolors()
            if len(facecolors) > 0:
                data["facecolors"] = facecolors.tolist()
        except:
            pass

        try:
            edgecolors = collection.get_edgecolors()
            if len(edgecolors) > 0:
                data["edgecolors"] = edgecolors.tolist()
        except:
            pass

    # Get offsets for PathCollection (scatter plots) - regular 2D
    elif isinstance(collection, PathCollection):
        offsets = collection.get_offsets()
        data["offsets"] = offsets.tolist() if len(offsets) > 0 else []
        data["sizes"] = collection.get_sizes().tolist() if len(collection.get_sizes()) > 0 else []

        # Get colors
        try:
            facecolors = collection.get_facecolors()
            if len(facecolors) > 0:
                data["facecolors"] = facecolors.tolist()
        except:
            pass

        try:
            edgecolors = collection.get_edgecolors()
            if len(edgecolors) > 0:
                data["edgecolors"] = edgecolors.tolist()
        except:
            pass

        data["linewidths"] = collection.get_linewidths().tolist() if len(collection.get_linewidths()) > 0 else []

    elif HAS_3D and isinstance(collection, Line3DCollection):
        # Line3DCollection has 3D segments
        try:
            if hasattr(collection, '_segments3d'):
                # _segments3d is a list of lists of (x,y,z) tuples
                segments3d = []
                for seg in collection._segments3d:
                    seg_data = []
                    for point in seg:
                        seg_data.append([float(point[0]), float(point[1]), float(point[2])])
                    segments3d.append(seg_data)
                data["segments3d"] = segments3d
        except:
            pass

        data["linewidths"] = collection.get_linewidths().tolist() if len(collection.get_linewidths()) > 0 else []

        try:
            colors = collection.get_colors()
            if len(colors) > 0:
                data["colors"] = colors.tolist()
        except:
            pass

    elif isinstance(collection, EventCollection):
        # EventCollection is a special LineCollection, handle it first
        segments = collection.get_segments()
        data["segments"] = [seg.tolist() for seg in segments]
        data["linewidths"] = collection.get_linewidths().tolist() if len(collection.get_linewidths()) > 0 else []

        # EventCollection-specific attributes
        try:
            data["lineoffsets"] = float(collection.get_lineoffsets()[0]) if hasattr(collection, 'get_lineoffsets') else 0
        except:
            data["lineoffsets"] = 0

        try:
            data["linelengths"] = float(collection.get_linelengths()[0]) if hasattr(collection, 'get_linelengths') else 1
        except:
            data["linelengths"] = 1

        try:
            colors = collection.get_colors()
            if len(colors) > 0:
                data["colors"] = colors.tolist()
        except:
            pass

    elif isinstance(collection, LineCollection):
        segments = collection.get_segments()
        data["segments"] = [seg.tolist() for seg in segments]
        data["linewidths"] = collection.get_linewidths().tolist() if len(collection.get_linewidths()) > 0 else []

        try:
            colors = collection.get_colors()
            if len(colors) > 0:
                data["colors"] = colors.tolist()
        except:
            pass

    elif isinstance(collection, Barbs):
        # Barbs is similar to Quiver but uses lowercase attributes
        try:
            data["x"] = collection.x.tolist()
            data["y"] = collection.y.tolist()
            data["u"] = collection.u.tolist()
            data["v"] = collection.v.tolist()
        except:
            pass

        try:
            facecolors = collection.get_facecolors()
            if len(facecolors) > 0:
                data["facecolors"] = facecolors.tolist()
        except:
            pass

        try:
            edgecolors = collection.get_edgecolors()
            if len(edgecolors) > 0:
                data["edgecolors"] = edgecolors.tolist()
        except:
            pass

    elif isinstance(collection, Quiver):
        # Quiver is a PolyCollection subclass, handle it first
        try:
            data["X"] = collection.X.tolist()
            data["Y"] = collection.Y.tolist()
            data["U"] = collection.U.tolist()
            data["V"] = collection.V.tolist()
        except:
            pass

        try:
            facecolors = collection.get_facecolors()
            if len(facecolors) > 0:
                data["facecolors"] = facecolors.tolist()
        except:
            pass

        try:
            edgecolors = collection.get_edgecolors()
            if len(edgecolors) > 0:
                data["edgecolors"] = edgecolors.tolist()
        except:
            pass

    elif isinstance(collection, PolyCollection):
        try:
            verts = collection.get_paths()
            data["verts"] = [[v.tolist() for v in path.vertices] for path in verts]
        except:
            pass

        try:
            facecolors = collection.get_facecolors()
            if len(facecolors) > 0:
                data["facecolors"] = facecolors.tolist()
        except:
            pass

        try:
            edgecolors = collection.get_edgecolors()
            if len(edgecolors) > 0:
                data["edgecolors"] = edgecolors.tolist()
        except:
            pass

    elif isinstance(collection, QuadMesh):
        try:
            # For QuadMesh, we need to store the coordinates and the array
            coordinates = collection.get_coordinates()
            if coordinates is not None:
                data["coordinates"] = coordinates.tolist()

            array = collection.get_array()
            if array is not None:
                data["array"] = array.tolist()
        except:
            pass

    # Colormap information
    try:
        cmap = collection.get_cmap()
        if cmap is not None:
            data["cmap"] = cmap.name
    except:
        pass

    try:
        norm = collection.norm
        if norm is not None:
            data["norm"] = {
                "vmin": float(norm.vmin) if norm.vmin is not None else None,
                "vmax": float(norm.vmax) if norm.vmax is not None else None,
                "clip": bool(norm.clip) if hasattr(norm, 'clip') else False,
            }
    except:
        pass

    return data


def _serialize_text(text: Text) -> Dict[str, Any]:
    """Serialize a Text object."""
    data = {
        "type": "Text" if not isinstance(text, Annotation) else "Annotation",
        "text": text.get_text(),
        "position": list(text.get_position()),
        "fontsize": float(text.get_fontsize()),
        "color": _serialize_color(text.get_color()),
        "horizontalalignment": text.get_horizontalalignment(),
        "verticalalignment": text.get_verticalalignment(),
        "rotation": float(text.get_rotation()),
        "alpha": text.get_alpha(),
        "visible": text.get_visible(),
        "zorder": float(text.get_zorder()),
        "family": text.get_family(),
        "style": text.get_style(),
        "weight": text.get_weight(),
    }

    if isinstance(text, Annotation):
        data["xy"] = list(text.xy)
        # Annotation uses xyann attribute, not xytext
        data["xytext"] = list(text.xyann) if hasattr(text, 'xyann') else None
        data["arrowprops"] = text.arrowprops

    return data


def _serialize_image(image: AxesImage) -> Dict[str, Any]:
    """Serialize an AxesImage object."""
    array = image.get_array()

    data = {
        "type": "AxesImage",
        "array": array.tolist() if array is not None else None,
        "extent": image.get_extent(),
        "alpha": image.get_alpha(),
        "visible": image.get_visible(),
        "zorder": float(image.get_zorder()),
        "interpolation": image.get_interpolation(),
    }

    # Colormap information
    try:
        cmap = image.get_cmap()
        if cmap is not None:
            data["cmap"] = cmap.name
    except:
        pass

    try:
        norm = image.norm
        if norm is not None:
            data["norm"] = {
                "vmin": float(norm.vmin) if norm.vmin is not None else None,
                "vmax": float(norm.vmax) if norm.vmax is not None else None,
            }
    except:
        pass

    return data


def _serialize_contour(contour: ContourSet) -> Dict[str, Any]:
    """Serialize a ContourSet object."""
    data = {
        "type": type(contour).__name__,
        "levels": contour.levels.tolist() if hasattr(contour.levels, 'tolist') else list(contour.levels),
        "alpha": contour.alpha,
        "visible": contour.get_visible() if hasattr(contour, 'get_visible') else True,
    }

    # Store the contour collections
    if hasattr(contour, 'collections'):
        data["collections"] = [_serialize_collection(c) for c in contour.collections]

    # Colormap
    try:
        if contour.cmap is not None:
            data["cmap"] = contour.cmap.name
    except:
        pass

    return data


def _serialize_quiver(quiver: Quiver) -> Dict[str, Any]:
    """Serialize a Quiver (arrow) plot."""
    data = {
        "type": "Quiver",
        "X": quiver.X.tolist() if hasattr(quiver.X, 'tolist') else None,
        "Y": quiver.Y.tolist() if hasattr(quiver.Y, 'tolist') else None,
        "U": quiver.U.tolist() if hasattr(quiver.U, 'tolist') else None,
        "V": quiver.V.tolist() if hasattr(quiver.V, 'tolist') else None,
        "alpha": quiver.get_alpha(),
        "visible": quiver.get_visible(),
        "zorder": float(quiver.get_zorder()),
    }

    # Colormap
    try:
        if quiver.cmap is not None:
            data["cmap"] = quiver.cmap.name
    except:
        pass

    return data


def _serialize_legend(legend: Legend) -> Dict[str, Any]:
    """Serialize a Legend object."""
    if legend is None:
        return None

    data = {
        "type": "Legend",
        "labels": [t.get_text() for t in legend.get_texts()],
        "loc": legend._loc if hasattr(legend, '_loc') else None,
        "frameon": legend.get_frame_on(),
        "shadow": legend.shadow,
        "framealpha": legend.get_frame().get_alpha() if legend.get_frame() else None,
        "facecolor": _serialize_color(legend.get_frame().get_facecolor() if legend.get_frame() else None),
        "edgecolor": _serialize_color(legend.get_frame().get_edgecolor() if legend.get_frame() else None),
        "title": legend.get_title().get_text() if legend.get_title() else None,
    }

    return data


def _serialize_axes(ax: Axes) -> Dict[str, Any]:
    """Serialize an Axes object."""
    # Check if 3D axes first
    is_3d = hasattr(ax, 'get_zlim')

    data = {
        "type": "Axes",
        "position": list(ax.get_position().bounds),
        "xlim": list(ax.get_xlim()),
        "ylim": list(ax.get_ylim()),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "title": ax.get_title(),
        "xscale": ax.get_xscale(),
        "yscale": ax.get_yscale(),
        "facecolor": _serialize_color(ax.get_facecolor()),
    }

    # Some properties differ for 2D vs 3D axes
    if not is_3d:
        data["aspect"] = ax.get_aspect()
        data["frame_on"] = ax.get_frame_on()
    else:
        data["aspect"] = "auto"
        data["frame_on"] = True

    data["is_3d"] = is_3d

    # Grid - check if grid is visible by looking at gridlines
    try:
        grid_visible = (len(ax.xaxis.get_gridlines()) > 0 and ax.xaxis.get_gridlines()[0].get_visible()) or \
                      (len(ax.yaxis.get_gridlines()) > 0 and ax.yaxis.get_gridlines()[0].get_visible())
    except:
        grid_visible = False

    data["grid"] = {
        "visible": grid_visible,
        "which": "both",
        "axis": "both",
    }

    # Ticks
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    data["xticks"] = xticks.tolist() if hasattr(xticks, 'tolist') else list(xticks)
    data["yticks"] = yticks.tolist() if hasattr(yticks, 'tolist') else list(yticks)
    data["xticklabels"] = [t.get_text() for t in ax.get_xticklabels()]
    data["yticklabels"] = [t.get_text() for t in ax.get_yticklabels()]

    # Spines
    data["spines"] = {}
    for spine_name, spine in ax.spines.items():
        data["spines"][spine_name] = {
            "visible": spine.get_visible(),
            "color": _serialize_color(spine.get_edgecolor()),
            "linewidth": float(spine.get_linewidth()),
        }

    # 3D specific properties
    if is_3d:
        data["zlim"] = list(ax.get_zlim())
        data["zlabel"] = ax.get_zlabel()
        zticks = ax.get_zticks()
        data["zticks"] = zticks.tolist() if hasattr(zticks, 'tolist') else list(zticks)
        data["zticklabels"] = [t.get_text() for t in ax.get_zticklabels()]

        # 3D view angle
        data["view_angle"] = {
            "elev": ax.elev,
            "azim": ax.azim,
        }

    # Serialize all artists
    data["lines"] = [_serialize_line2d(line) for line in ax.get_lines()]
    data["patches"] = [_serialize_patch(patch) for patch in ax.patches]
    data["collections"] = [_serialize_collection(col) for col in ax.collections]
    data["texts"] = [_serialize_text(text) for text in ax.texts]
    data["images"] = [_serialize_image(img) for img in ax.get_images()]

    # Special artists
    data["artists"] = []
    for artist in ax.artists:
        if isinstance(artist, Line2D):
            data["artists"].append(_serialize_line2d(artist))
        elif isinstance(artist, Patch):
            data["artists"].append(_serialize_patch(artist))
        elif isinstance(artist, Text):
            data["artists"].append(_serialize_text(artist))

    # Legend
    legend = ax.get_legend()
    data["legend"] = _serialize_legend(legend)

    return data


def _serialize_figure(fig: Figure) -> Dict[str, Any]:
    """Serialize a Figure object."""
    data = {
        "type": "Figure",
        "figsize": list(fig.get_size_inches()),
        "dpi": float(fig.dpi),
        "facecolor": _serialize_color(fig.get_facecolor()),
        "edgecolor": _serialize_color(fig.get_edgecolor()),
        "frameon": fig.frameon,
        "tight_layout": fig.get_tight_layout(),
        "constrained_layout": fig.get_constrained_layout(),
    }

    # Serialize all axes
    data["axes"] = [_serialize_axes(ax) for ax in fig.get_axes()]

    # Suptitle
    suptitle = fig._suptitle
    if suptitle is not None:
        data["suptitle"] = {
            "text": suptitle.get_text(),
            "fontsize": float(suptitle.get_fontsize()),
            "fontweight": suptitle.get_weight(),
        }
    else:
        data["suptitle"] = None

    # Figure texts (not in axes)
    data["texts"] = [_serialize_text(text) for text in fig.texts if text != suptitle]

    return data


def save_figure(fig: Figure, filepath: str) -> None:
    """
    Save a matplotlib figure to a JSON file by serializing all its data.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    filepath : str
        Path to save the JSON file

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pltsave import save_figure
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> save_figure(fig, 'my_figure.json')
    """
    data = _serialize_figure(fig)

    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
