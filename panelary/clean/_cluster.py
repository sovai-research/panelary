"""Connected components over candidate edges: a vectorised numpy union-find.

The near-duplicate pipeline ends in a graph problem: rows are nodes, verified
near-duplicate pairs are edges, and a *cluster* is a connected component. This
module solves it with no Python loop over rows or edges.

The algorithm is hook-and-compress (the Shiloach-Vishkin family): every round
hooks the larger root of each edge onto the smaller one with
``np.minimum.at``, then pointer-jumps until every node points at its root. The
invariant ``parent[x] <= x`` holds throughout, so there are no cycles, and a
component's root is its **smallest node id** -- which makes the labelling
canonical and deterministic, independent of edge order.
"""

from __future__ import annotations

import numpy as np

__all__ = ["connected_components", "later_endpoints"]


def _compress(parent: np.ndarray) -> np.ndarray:
    """Pointer-jump until every node points directly at its root."""
    while True:
        grand = parent[parent]
        if np.array_equal(grand, parent):
            return parent
        parent = grand


def connected_components(
    n: int, edges_i: np.ndarray, edges_j: np.ndarray
) -> np.ndarray:
    """Label the connected components of an undirected graph on ``range(n)``.

    Parameters
    ----------
    n : int
        Number of nodes. Node ids are ``0 .. n - 1``.
    edges_i, edges_j : numpy.ndarray
        Integer endpoint arrays of equal length. Self-loops and repeated edges
        are harmless.

    Returns
    -------
    numpy.ndarray
        ``int64`` array of length ``n``: each node's component label, which is
        the **smallest node id** in its component. Isolated nodes label
        themselves.

    Raises
    ------
    ValueError
        If the endpoint arrays differ in length or reference a node outside
        ``range(n)``.

    Examples
    --------
    >>> connected_components(5, np.array([0, 3]), np.array([2, 4])).tolist()
    [0, 1, 0, 3, 3]
    """
    ei = np.asarray(edges_i, dtype=np.int64).ravel()
    ej = np.asarray(edges_j, dtype=np.int64).ravel()
    if ei.shape != ej.shape:
        raise ValueError(
            f"edge endpoint arrays differ in length ({ei.size} vs {ej.size})."
        )
    parent = np.arange(n, dtype=np.int64)
    if ei.size == 0:
        return parent
    if min(int(ei.min()), int(ej.min())) < 0 or max(int(ei.max()), int(ej.max())) >= n:
        raise ValueError(f"edge endpoints must lie in [0, {n}).")
    while True:
        ri = parent[ei]
        rj = parent[ej]
        pending = ri != rj
        if not pending.any():
            return parent
        lo = np.minimum(ri[pending], rj[pending])
        hi = np.maximum(ri[pending], rj[pending])
        # Hook: each root adopts the smallest root it is linked to this round.
        np.minimum.at(parent, hi, lo)
        parent = _compress(parent)


def later_endpoints(
    n: int, edges_i: np.ndarray, edges_j: np.ndarray, order: np.ndarray
) -> np.ndarray:
    """Mark every node that has an edge to a node **earlier** in ``order``.

    This is the point-in-time drop rule: a row is a duplicate iff it
    near-duplicates some row that came before it. It is *pairwise*, so unlike
    connected components it can never be changed by rows that arrive later.

    Parameters
    ----------
    n : int
        Number of nodes.
    edges_i, edges_j : numpy.ndarray
        Integer endpoint arrays.
    order : numpy.ndarray
        Length-``n`` array of distinct ranks; lower means earlier.

    Returns
    -------
    numpy.ndarray
        Boolean mask of length ``n``.
    """
    out = np.zeros(n, dtype=bool)
    ei = np.asarray(edges_i, dtype=np.int64).ravel()
    ej = np.asarray(edges_j, dtype=np.int64).ravel()
    if ei.size == 0:
        return out
    rank = np.asarray(order)
    later = np.where(rank[ei] > rank[ej], ei, ej)
    distinct = ei != ej
    out[later[distinct]] = True
    return out
