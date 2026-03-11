"""
pathfinding.py - Grid-Based Pathfinding Algorithms
====================================================
A* and BFS pathfinding on a 2D grid with obstacles.

The grid is a 2D list of CellState values.  Each generator yields one
PathStep per cell expansion so the GUI can animate exploration.

Key A* concepts:
    g(n) = exact cost from start to n (number of steps)
    h(n) = heuristic estimate of cost from n to goal
    f(n) = g(n) + h(n)                    ← priority key

We use Manhattan distance as the heuristic:
    h(n) = |n.row - goal.row| + |n.col - goal.col|

Manhattan distance is *admissible* (never overestimates) on a 4-connected
grid, so A* will always find the optimal path.
"""

import heapq
from typing import Generator, List, Optional, Tuple

from utils import CellState, PathStep


# Type alias for a 2-D grid
Grid = List[List[CellState]]


# ── Grid helpers ─────────────────────────────────────────────────────────────

def make_empty_grid(rows: int, cols: int) -> Grid:
    """Return a fresh grid filled with EMPTY cells."""
    return [[CellState.EMPTY] * cols for _ in range(rows)]


def neighbours(row: int, col: int, rows: int, cols: int,
               diagonal: bool = False) -> List[Tuple[int, int]]:
    """
    Return valid (row, col) neighbours for a cell.
    4-connected by default; 8-connected if diagonal=True.
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    result = []
    for dr, dc in directions:
        r2, c2 = row + dr, col + dc
        if 0 <= r2 < rows and 0 <= c2 < cols:
            result.append((r2, c2))
    return result


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance heuristic — admissible on a 4-connected grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from: dict,
                     start: Tuple[int, int],
                     end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Trace parent pointers back from end → start."""
    path = []
    node: Optional[Tuple[int, int]] = end
    while node and node != start:
        path.append(node)
        node = came_from.get(node)
    if node == start:
        path.append(start)
    return list(reversed(path))


# ── BFS on grid ──────────────────────────────────────────────────────────────

def grid_bfs(grid: Grid,
             start: Tuple[int, int],
             end: Tuple[int, int]) -> Generator[PathStep, None, None]:
    """
    BFS on a 2-D grid — O(rows × cols)
    -------------------------------------
    Explores cells in order of their hop-distance from start.
    Guaranteed to find the shortest path (fewest steps) in an
    unweighted grid.
    
    Yields one PathStep per cell dequeued so the GUI shows exploration.
    """
    rows = len(grid)
    cols = len(grid[0])

    from collections import deque
    queue     = deque([start])
    visited   = {start}
    came_from = {start: None}

    yield PathStep(
        visited=set(visited), frontier={start},
        message=f"BFS: Starting at {start}"
    )

    while queue:
        cell = queue.popleft()
        r, c = cell

        if cell == end:
            path = reconstruct_path(came_from, start, end)
            yield PathStep(
                visited=set(visited), frontier=set(),
                path=path, message=f"BFS: Goal reached! Path length = {len(path)-1}"
            )
            return

        for nr, nc in neighbours(r, c, rows, cols):
            nbr = (nr, nc)
            state = grid[nr][nc]
            if nbr not in visited and state != CellState.WALL:
                visited.add(nbr)
                queue.append(nbr)
                came_from[nbr] = cell

                yield PathStep(
                    visited=set(visited),
                    frontier=set(queue),
                    active_edge=(cell, nbr),
                    message=f"BFS: Visited {nbr} from {cell}"
                )

    yield PathStep(
        visited=set(visited), frontier=set(),
        message="BFS: No path found!"
    )


# ── A* on grid ───────────────────────────────────────────────────────────────

def astar(grid: Grid,
          start: Tuple[int, int],
          end: Tuple[int, int]) -> Generator[PathStep, None, None]:
    """
    A* Pathfinding — optimal + heuristically guided
    -------------------------------------------------
    Priority queue ordered by f = g + h.
    
    Compared to BFS:
      • BFS explores rings of equal hop-distance (concentric circles).
      • A* focuses exploration toward the goal → visits far fewer cells.
    
    With Manhattan distance heuristic on a 4-grid, A* is optimal and
    efficient.

    open_set  = cells to be evaluated (min-heap on f)
    closed_set = cells already evaluated
    g[n]      = cheapest known path from start to n
    """
    rows = len(grid)
    cols = len(grid[0])

    g_score: dict = {start: 0}
    f_score: dict = {start: manhattan(start, end)}

    # Heap entries: (f_score, tie_breaker, cell)
    counter    = 0
    open_heap  = [(f_score[start], counter, start)]
    open_set   = {start}
    closed_set: set = set()
    came_from: dict = {}

    yield PathStep(
        visited=set(closed_set),
        frontier=set(open_set),
        message=f"A*: Start={start}, Goal={end}, h={manhattan(start,end)}"
    )

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current not in open_set:
            continue   # Stale heap entry

        open_set.discard(current)
        closed_set.add(current)

        if current == end:
            path = reconstruct_path(came_from, start, end)
            yield PathStep(
                visited=set(closed_set),
                frontier=set(open_set),
                path=path,
                message=f"A*: Goal reached! Path cost = {g_score[end]}"
            )
            return

        yield PathStep(
            visited=set(closed_set),
            frontier=set(open_set),
            active_edge=(),
            message=f"A*: Expand {current}, g={g_score[current]}, "
                    f"f={f_score.get(current, '?')}"
        )

        r, c = current
        for nr, nc in neighbours(r, c, rows, cols):
            nbr = (nr, nc)
            if nbr in closed_set or grid[nr][nc] == CellState.WALL:
                continue

            tentative_g = g_score[current] + 1   # unit-cost grid

            if tentative_g < g_score.get(nbr, float("inf")):
                came_from[nbr] = current
                g_score[nbr]   = tentative_g
                f_score[nbr]   = tentative_g + manhattan(nbr, end)
                open_set.add(nbr)
                counter += 1
                heapq.heappush(open_heap, (f_score[nbr], counter, nbr))

                yield PathStep(
                    visited=set(closed_set),
                    frontier=set(open_set),
                    active_edge=(current, nbr),
                    message=f"A*: Update {nbr} — g={tentative_g}, "
                            f"h={manhattan(nbr,end)}, f={f_score[nbr]}"
                )

    yield PathStep(
        visited=set(closed_set), frontier=set(),
        message="A*: No path found!"
    )
