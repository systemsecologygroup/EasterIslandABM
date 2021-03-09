import matplotlib as mpl
import copy
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib.patches import Patch, Rectangle


lake_color = (3/255, 169/255, 244/255, 1)
cmap_lake = mpl.colors.ListedColormap([(0, 0, 0, 0), lake_color])

cmap_fp = copy.copy(mpl.cm.get_cmap("autumn_r"))
cmap_fp.set_under("white")

green = mpl.cm.get_cmap("Greens")(np.linspace(0, 1, int(256*1.5)))
cmap_trees = LinearSegmentedColormap.from_list("greenhalf", green[:256])


def plot_map(map, var, label, cmap_x, vmin, vmax, title):
    w_ax = 5.6572
    h_ax = 4.1652
    xmax_Springer = 7.8
    cbarw = 1.2
    fig = plt.figure(figsize=((xmax_Springer) * 0.393701, (xmax_Springer - cbarw) * h_ax / w_ax * 0.393701))
    #fig = plt.figure()
    ax = fig.add_subplot(111, fc=(236 / 255, 239 / 255, 236 / 255), frame_on=True)
    for s in ax.spines.keys():
        ax.spines[s].set_visible(False)
    ax.set_xlim(1, 24)
    ax.set_ylim(1, 23 * h_ax / w_ax + 1)
    ax.set_aspect("equal")

    fontsize = 7
    t = 800
    ax.invert_yaxis()

    # Create cax that includes the colorbar
    aspect = 20
    pad_fraction = 0.5
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1. / aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)

    # Plot Variable
    x = np.zeros_like(map.triobject.mask, dtype=np.float32)
    x[map.inds_map] = var
    trpc = ax.tripcolor(map.triobject, facecolors=x, cmap=cmap_x, edgecolors="none", vmin=vmin, vmax=vmax,
                        snap=True)
    plt.colorbar(trpc, cax=cax, label=label,
                 extend="min" if cmap_x == cmap_fp else "neither")  # , pad=0.02, fraction=0.1, shrink=ax.get_position().height)

    # Plot Lakes
    lakes = np.zeros_like(map.triobject.mask, dtype=np.int8)
    lakes[map.inds_map[map.water_cells_map]] = 1
    ax.tripcolor(map.triobject, facecolors=lakes, cmap=cmap_lake, edgecolors="none", snap=True, alpha=None)

    ax.set_title(" " + str(t) + "$\,$A.D." + r"  ", loc="right", fontsize=11, fontweight="normal", y=0.011)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    # Lake and scale bar
    scale = Rectangle((10, 15.9), 5, 0.3, color="k", fill=True)
    ax.add_artist(scale)
    ax.text(12.5, 17.5, "5 km", ha="center", va="bottom", fontsize=fontsize)
    ax.text(22, 4, "Lake", ha="center", va="bottom", fontsize=fontsize)
    l = Rectangle((21, 1.5), 2, 1, color=lake_color, fill=True)
    ax.add_artist(l)

    fig.tight_layout(pad=0)
    plt.savefig("Map/"+title+".eps", bbox_inches="tight", pad_inches=0)

    return