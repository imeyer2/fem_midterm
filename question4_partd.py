"""
Chapter 3: Piecewise Polynomial Approximations in 2D
Self-contained L2 projector with a Python 'initmesh' for [0,1]^2 \ [0.5,1]^2
"""

import numpy as np
from matplotlib.tri import Triangulation

def PolyArea(x: np.ndarray, y: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def MassAssembler2D(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Assemble the global P1 mass matrix on triangles.
    p: shape (2, num_nodes), t: shape (3, num_elements) with 0-based indices.
    """
    assert p.shape[0] == 2, "p must have shape (2, num_nodes)"
    assert t.shape[0] == 3, "t must have shape (3, num_elements)"
    num_points = p.shape[1]
    num_elements = t.shape[1]

    M = np.zeros((num_points, num_points))
    for K in range(num_elements):
        loc2glb = t[:, K]
        x = p[0, loc2glb]
        y = p[1, loc2glb]
        area = PolyArea(x, y)
        MK = (area / 12.0) * np.array([[2, 1, 1],
                                       [1, 2, 1],
                                       [1, 1, 2]], dtype=float)
        M[np.ix_(loc2glb, loc2glb)] += MK
    return M

def LoadAssembler2D(p: np.ndarray, t: np.ndarray, f: callable) -> np.ndarray:
    """
    Assemble load vector for L2 projection: use vertex-based lumped quadrature (area/3 * f at vertices).
    """
    assert p.shape[0] == 2 and t.shape[0] == 3
    num_points = p.shape[1]
    num_elements = t.shape[1]
    b = np.zeros(num_points)
    for K in range(num_elements):
        loc2glb = t[:, K]
        x = p[0, loc2glb]
        y = p[1, loc2glb]
        area = PolyArea(x, y)
        bK = np.array([f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2])], dtype=float) * (area / 3.0)
        b[loc2glb] += bK
    return b

def L2Projector2D(p: np.ndarray, t: np.ndarray, f: callable) -> np.ndarray:
    """
    Compute the P1 nodal L2 projection of f over a triangulation.
    """
    # Ensure 0-based indexing for t if someone passed MATLAB-style 1-based
    if t.max() >= p.shape[1]:
        raise ValueError("t has indices >= number of points. Check your mesh.")
    M = MassAssembler2D(p, t)
    b = LoadAssembler2D(p, t, f)
    return np.linalg.solve(M, b)

def initmesh(g, hmax=0.1, hole=None):
    """
    Python analogue of MATLAB PDE Toolbox 'initmesh'

    Parameters
    ----------
    g : tuple (xmin, xmax, ymin, ymax)
        Outer rectangle.
    hmax : float
        Target max edge length (grid spacing heuristic).
    hole : tuple or None
        If provided, (xmin, xmax, ymin, ymax) rectangle to excise.

    Returns
    -------
    p : ndarray shape (2, N)
        Node coordinates.
    t : ndarray shape (3, NT)
        Triangles (0-based indices).
    e : ndarray shape (2, NE)
        Unique edges (0-based). Boundary-only edges can be derived if needed.
    """
    xmin, xmax, ymin, ymax = g
    nx = max(3, int(np.ceil((xmax - xmin) / hmax)) + 1)
    ny = max(3, int(np.ceil((ymax - ymin) / hmax)) + 1)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)
    P = np.vstack((xx.ravel(), yy.ravel()))

    tri = Triangulation(P[0], P[1])
    T = tri.triangles.copy()# shape (ntri, 3)

    # Filter out triangles whose centroid lies in the hole rectangle
    if hole is not None:
        hx0, hx1, hy0, hy1 = hole
        centroids = P[:, T].mean(axis=2)  # shape (2, ntri)
        in_hole = (
            (centroids[0] >= hx0) & (centroids[0] <= hx1) &
            (centroids[1] >= hy0) & (centroids[1] <= hy1)
        )
        T = T[~in_hole]

    # Remove isolated nodes and reindex triangles to compact node set
    used_nodes = np.unique(T.ravel())
    old_to_new = -np.ones(P.shape[1], dtype=int)
    old_to_new[used_nodes] = np.arange(used_nodes.size, dtype=int)

    p = P[:, used_nodes]
    t = old_to_new[T].T

    # Build unique edges from triangles
    edges = set()
    for tri_nodes in t.T:
        i, j, k = int(tri_nodes[0]), int(tri_nodes[1]), int(tri_nodes[2])
        for a, b in ((i, j), (j, k), (k, i)):
            edges.add(tuple(sorted((a, b))))
    e = np.array(list(edges), dtype=int).T if edges else np.zeros((2, 0), dtype=int)

    return p, t, e


if __name__ == "__main__":
    # Domain [0,1]^2 minus the inner square [0.5,1]^2
    outer = (0.0, 1.0, 0.0, 1.0)
    inner_hole = (0.5, 1.0, 0.5, 1.0)

    p, t, e = initmesh(outer, hmax=0.1, hole=inner_hole)

    # Test function
    f = lambda x, y: 1.0 - x**2 - y**2

    Pf = L2Projector2D(p, t, f)
    print("Nodes:", p.shape[1], "Triangles:", t.shape[1], "Edges:", e.shape[1])
    print("L2 projection Pf (first 10 entries):", Pf[:10])

    # visualize with matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation as mTri

    tri_plot = mTri(p[0], p[1], t.T)

    f_nodes = 1.0 - p[0]**2 - p[1]**2

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    im0 = axs[0].tripcolor(tri_plot, f_nodes, shading="gouraud")
    axs[0].triplot(tri_plot, color="k", lw=0.2, alpha=0.4)
    axs[0].set_title("Exact $f(x,y)=1-x^2-y^2$ at nodes")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("y"); axs[0].axis("equal")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].tripcolor(tri_plot, Pf, shading="gouraud")
    axs[1].triplot(tri_plot, color="k", lw=0.2, alpha=0.4)
    axs[1].set_title("Projected $P_h f$")
    axs[1].set_xlabel("x"); axs[1].set_ylabel("y"); axs[1].axis("equal")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.savefig("question4.png", dpi=150)
