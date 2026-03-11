"""
graph_algorithms.py - Graph Algorithm Step Generators
======================================================
BFS, DFS, and Dijkstra are implemented as generators that yield one
PathStep per visited/frontier update so the GUI can animate each step.

The graph is represented as an adjacency list stored on a list of
GraphNode / GraphEdge objects (see utils.py).
"""

import heapq
import math
import random
from collections import deque
from typing import Dict, Generator, List, Tuple

from utils import GraphEdge, GraphNode, PathStep


# ── Graph factory ───────────────────────────────────────────────────────────

def make_sample_graph(width: int = 700, height: int = 480,
                      n_nodes: int = 12) -> Tuple[List[GraphNode], List[GraphEdge]]:
    """
    Generate a random connected weighted graph for demonstration.
    
    Nodes are placed on a grid with slight random jitter; edges are added
    to ensure connectivity plus a few cross-edges for variety.
    """
    random.seed(42)   # reproducible layout
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []

    # Place nodes in a grid-ish layout
    cols = math.ceil(math.sqrt(n_nodes))
    for i in range(n_nodes):
        row = i // cols
        col = i % cols
        x = 80 + col * ((width - 160) / (cols - 1)) + random.randint(-20, 20)
        y = 70 + row * ((height - 140) / (math.ceil(n_nodes / cols) - 1 or 1)) + random.randint(-20, 20)
        nodes.append(GraphNode(node_id=i, x=x, y=y, label=str(i)))

    # Build a spanning tree first (guarantees connectivity)
    node_ids = list(range(n_nodes))
    random.shuffle(node_ids)
    in_tree = {node_ids[0]}
    for nid in node_ids[1:]:
        src = random.choice(list(in_tree))
        w   = round(math.dist((nodes[src].x, nodes[src].y),
                               (nodes[nid].x, nodes[nid].y)) / 40, 1)
        edges.append(GraphEdge(src=src, dst=nid, weight=w))
        edges.append(GraphEdge(src=nid, dst=src, weight=w))
        in_tree.add(nid)

    # Add a handful of extra edges
    extra = n_nodes // 2
    attempts = 0
    while extra > 0 and attempts < 200:
        attempts += 1
        u, v = random.sample(range(n_nodes), 2)
        already = any((e.src == u and e.dst == v) for e in edges)
        if not already:
            w = round(math.dist((nodes[u].x, nodes[u].y),
                                 (nodes[v].x, nodes[v].y)) / 40, 1)
            edges.append(GraphEdge(src=u, dst=v, weight=w))
            edges.append(GraphEdge(src=v, dst=u, weight=w))
            extra -= 1

    return nodes, edges


def adjacency(edges: List[GraphEdge]) -> Dict[int, List[Tuple[int, float]]]:
    """Build an adjacency list: node_id → [(neighbour, weight), ...]."""
    adj: Dict[int, List[Tuple[int, float]]] = {}
    for e in edges:
        adj.setdefault(e.src, []).append((e.dst, e.weight))
    return adj


# ── BFS ─────────────────────────────────────────────────────────────────────

def bfs(nodes: List[GraphNode], edges: List[GraphEdge],
        start: int = 0) -> Generator[PathStep, None, None]:
    """
    Breadth-First Search — O(V + E)
    ---------------------------------
    Uses a FIFO queue to visit nodes level by level.
    Shortest hop-count path from source in an unweighted graph.
    
    Yields one PathStep after each dequeue + neighbour-enqueue cycle.
    """
    adj = adjacency(edges)
    visited  = set()
    frontier = deque([start])
    visited.add(start)
    parent  = {start: None}
    active_edge = ()

    yield PathStep(visited=set(visited), frontier=set(frontier),
                   message=f"BFS: Start at node {start}")

    while frontier:
        node = frontier.popleft()

        for neighbour, _ in adj.get(node, []):
            active_edge = (node, neighbour)
            if neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
                parent[neighbour] = node

                yield PathStep(
                    visited=set(visited),
                    frontier=set(frontier),
                    active_edge=active_edge,
                    message=f"BFS: Visiting {node} → enqueue neighbour {neighbour}"
                )
            else:
                yield PathStep(
                    visited=set(visited),
                    frontier=set(frontier),
                    active_edge=active_edge,
                    message=f"BFS: {neighbour} already visited — skip"
                )

    # Reconstruct path to the last-visited node (farthest from start)
    path = _reconstruct(parent, start, max(visited))
    yield PathStep(visited=set(visited), frontier=set(),
                   path=path, message="BFS complete — path highlighted")


# ── DFS ─────────────────────────────────────────────────────────────────────

def dfs(nodes: List[GraphNode], edges: List[GraphEdge],
        start: int = 0) -> Generator[PathStep, None, None]:
    """
    Depth-First Search — O(V + E)
    --------------------------------
    Uses a LIFO stack (iterative, not recursive) to explore as deep as
    possible before backtracking.
    
    Useful for: cycle detection, topological sort, maze generation.
    """
    adj = adjacency(edges)
    visited  = set()
    stack    = [start]
    parent   = {start: None}
    active_edge = ()

    yield PathStep(visited=set(), frontier={start},
                   message=f"DFS: Push node {start} onto stack")

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        yield PathStep(
            visited=set(visited),
            frontier=set(stack),
            active_edge=active_edge,
            message=f"DFS: Pop and visit node {node}"
        )

        for neighbour, _ in reversed(adj.get(node, [])):
            if neighbour not in visited:
                stack.append(neighbour)
                parent.setdefault(neighbour, node)
                active_edge = (node, neighbour)
                yield PathStep(
                    visited=set(visited),
                    frontier=set(stack),
                    active_edge=active_edge,
                    message=f"DFS: Push neighbour {neighbour} onto stack"
                )

    path = _reconstruct(parent, start, max(visited))
    yield PathStep(visited=set(visited), frontier=set(),
                   path=path, message="DFS complete — path highlighted")


# ── Dijkstra ─────────────────────────────────────────────────────────────────

def dijkstra(nodes: List[GraphNode], edges: List[GraphEdge],
             start: int = 0) -> Generator[PathStep, None, None]:
    """
    Dijkstra's Shortest Path — O((V + E) log V)
    ----------------------------------------------
    Uses a min-heap priority queue to always expand the node with the
    smallest known distance.
    
    Works on graphs with non-negative edge weights.
    Classic algorithm behind GPS navigation and network routing.
    """
    adj = adjacency(edges)
    n   = len(nodes)

    dist   = {i: float("inf") for i in range(n)}
    dist[start] = 0.0
    parent = {start: None}
    heap   = [(0.0, start)]    # (distance, node_id)
    visited  = set()
    frontier = {start}

    yield PathStep(
        visited=set(), frontier=frontier,
        distances=dict(dist),
        message=f"Dijkstra: Start at node {start}, dist=0"
    )

    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        frontier.discard(node)

        yield PathStep(
            visited=set(visited), frontier=set(frontier),
            distances=dict(dist), active_edge=(),
            message=f"Dijkstra: Extract min — node {node} (dist={d:.1f})"
        )

        for neighbour, weight in adj.get(node, []):
            if neighbour in visited:
                continue
            new_dist = d + weight
            yield PathStep(
                visited=set(visited), frontier=set(frontier),
                distances=dict(dist), active_edge=(node, neighbour),
                message=f"Relax edge {node}→{neighbour}: "
                        f"{d:.1f} + {weight:.1f} = {new_dist:.1f} "
                        f"(current best: {dist[neighbour]:.1f})"
            )
            if new_dist < dist[neighbour]:
                dist[neighbour] = new_dist
                parent[neighbour] = node
                heapq.heappush(heap, (new_dist, neighbour))
                frontier.add(neighbour)

    target = max(range(n), key=lambda i: dist[i] if dist[i] != float("inf") else -1)
    path   = _reconstruct(parent, start, target)
    yield PathStep(
        visited=set(visited), frontier=set(),
        distances=dict(dist), path=path,
        message=f"Dijkstra complete — shortest path to node {target} highlighted"
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _reconstruct(parent: dict, start: int, end: int) -> List[int]:
    """Walk the parent map backwards from end → start to get the path."""
    path = []
    node: int | None = end
    while node is not None:
        path.append(node)
        node = parent.get(node)
        if node == start:
            path.append(start)
            break
    return list(reversed(path))
