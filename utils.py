"""
utils.py - Shared constants, color palette, and helper utilities
=================================================================
All colors, timing constants, and small helper functions live here so
the rest of the codebase imports from a single source of truth.
"""

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional


# ── Color Palette ─────────────────────────────────────────────────────────
# A dark "terminal-meets-neon" theme. All hex strings, used by QPainter.

class Colors:
    # Backgrounds
    BG_DARK       = "#0D1117"   # Main window background
    BG_PANEL      = "#161B22"   # Side panel / card background
    BG_WIDGET     = "#1C2128"   # Widget interior
    BG_HOVER      = "#21262D"   # Button hover

    # Bars / nodes (default state)
    BAR_DEFAULT   = "#3D8EF0"   # Calm blue — unsorted bar
    BAR_COMPARE   = "#F0A030"   # Amber — currently compared
    BAR_SWAP      = "#FF4560"   # Red-pink — being swapped
    BAR_SORTED    = "#20C997"   # Teal-green — confirmed sorted
    BAR_PIVOT     = "#BF7FFF"   # Purple — quicksort pivot

    # Graph / pathfinding nodes
    NODE_DEFAULT  = "#2D3748"   # Dark grey node
    NODE_VISITED  = "#3D8EF0"   # Blue — BFS/DFS visited
    NODE_FRONTIER = "#F0A030"   # Amber — in queue/stack
    NODE_PATH     = "#20C997"   # Green — final path
    NODE_START    = "#20C997"   # Green start
    NODE_END      = "#FF4560"   # Red end
    NODE_WALL     = "#0D1117"   # Near-black wall
    NODE_OPEN     = "#BF7FFF"   # Purple — A* open set
    NODE_CLOSED   = "#3D8EF0"   # Blue — A* closed set
    EDGE_DEFAULT  = "#3A4050"
    EDGE_ACTIVE   = "#F0A030"

    # UI chrome
    TEXT_PRIMARY  = "#E6EDF3"
    TEXT_MUTED    = "#7D8590"
    ACCENT        = "#3D8EF0"
    BORDER        = "#30363D"
    SLIDER_GROOVE = "#30363D"
    SLIDER_HANDLE = "#3D8EF0"


# ── Timing ─────────────────────────────────────────────────────────────────

MIN_DELAY_MS  = 10    # Fastest speed
MAX_DELAY_MS  = 800   # Slowest speed
DEFAULT_DELAY = 200   # Starting speed


# ── Sort-bar state enum ─────────────────────────────────────────────────────

class BarState(Enum):
    DEFAULT  = auto()
    COMPARE  = auto()
    SWAP     = auto()
    SORTED   = auto()
    PIVOT    = auto()


# ── Grid cell state enum ────────────────────────────────────────────────────

class CellState(Enum):
    EMPTY    = auto()
    WALL     = auto()
    START    = auto()
    END      = auto()
    VISITED  = auto()
    FRONTIER = auto()
    PATH     = auto()
    OPEN     = auto()    # A* open set
    CLOSED   = auto()    # A* closed set


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class SortStep:
    """
    Represents one atomic step produced by a sorting algorithm generator.
    
    Fields:
        array:      Full array snapshot at this step
        highlights: Dict mapping index → BarState for coloring
        message:    Human-readable description of what just happened
    """
    array:      List[int]
    highlights: dict = field(default_factory=dict)   # {index: BarState}
    message:    str  = ""


@dataclass
class GraphNode:
    """A node in the graph visualizer."""
    node_id: int
    x: float
    y: float
    label: str = ""
    state: str = "default"    # "default" | "visited" | "frontier" | "path"
    distance: float = float("inf")
    

@dataclass
class GraphEdge:
    """A weighted directed edge."""
    src: int
    dst: int
    weight: float = 1.0
    active: bool = False


@dataclass  
class PathStep:
    """One step in a graph/pathfinding algorithm."""
    visited: set     = field(default_factory=set)
    frontier: set    = field(default_factory=set)
    path: list       = field(default_factory=list)
    active_edge: tuple = field(default_factory=tuple)
    distances: dict  = field(default_factory=dict)
    message: str     = ""


# ── Helper functions ────────────────────────────────────────────────────────

def generate_random_array(n: int = 50, lo: int = 10, hi: int = 400) -> List[int]:
    """Generate a shuffled list of n unique integers in [lo, hi]."""
    values = random.sample(range(lo, hi), min(n, hi - lo))
    random.shuffle(values)
    return values


def lerp_color(hex_a: str, hex_b: str, t: float) -> str:
    """
    Linear interpolation between two hex colors.
    t=0 → hex_a, t=1 → hex_b. Used for smooth gradient highlights.
    """
    def parse(h: str) -> Tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    r1, g1, b1 = parse(hex_a)
    r2, g2, b2 = parse(hex_b)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def bar_color(state: BarState) -> str:
    """Map a BarState enum value to a hex color string."""
    return {
        BarState.DEFAULT : Colors.BAR_DEFAULT,
        BarState.COMPARE : Colors.BAR_COMPARE,
        BarState.SWAP    : Colors.BAR_SWAP,
        BarState.SORTED  : Colors.BAR_SORTED,
        BarState.PIVOT   : Colors.BAR_PIVOT,
    }.get(state, Colors.BAR_DEFAULT)


def cell_color(state: CellState) -> str:
    """Map a CellState to a hex color string."""
    return {
        CellState.EMPTY    : Colors.BG_WIDGET,
        CellState.WALL     : Colors.NODE_WALL,
        CellState.START    : Colors.NODE_START,
        CellState.END      : Colors.NODE_END,
        CellState.VISITED  : Colors.NODE_VISITED,
        CellState.FRONTIER : Colors.NODE_FRONTIER,
        CellState.PATH     : Colors.NODE_PATH,
        CellState.OPEN     : Colors.NODE_OPEN,
        CellState.CLOSED   : Colors.NODE_CLOSED,
    }.get(state, Colors.BG_WIDGET)


ALGORITHM_DESCRIPTIONS = {
    # Sorting
    "Bubble Sort": (
        "Bubble Sort",
        "O(n²) time  ·  O(1) space  ·  Stable",
        "Repeatedly steps through the list, compares adjacent elements and "
        "swaps them if they are in the wrong order. After each full pass, "
        "the largest unsorted element 'bubbles up' to its correct position.\n\n"
        "Best case O(n) when already sorted (with early-exit optimization)."
    ),
    "Merge Sort": (
        "Merge Sort",
        "O(n log n) time  ·  O(n) space  ·  Stable",
        "A divide-and-conquer algorithm that splits the array in half "
        "recursively until single elements remain, then merges them back "
        "in sorted order.\n\n"
        "Guaranteed O(n log n) in all cases — the gold standard for "
        "comparison-based sorting."
    ),
    "Quick Sort": (
        "Quick Sort",
        "O(n log n) avg  ·  O(log n) space  ·  Unstable",
        "Picks a pivot element, partitions the array so all smaller elements "
        "are to the left and larger to the right, then recurses on each "
        "partition.\n\n"
        "Fastest in practice due to excellent cache behaviour. O(n²) worst "
        "case with bad pivot selection."
    ),
    "Heap Sort": (
        "Heap Sort",
        "O(n log n) time  ·  O(1) space  ·  Unstable",
        "Builds a max-heap from the array, then repeatedly extracts the "
        "maximum element and places it at the end.\n\n"
        "In-place with guaranteed O(n log n) — combines the best of merge "
        "sort (complexity) and insertion sort (memory)."
    ),
    # Graph
    "BFS": (
        "Breadth-First Search",
        "O(V + E) time  ·  O(V) space",
        "Explores all neighbours of the current node before moving deeper. "
        "Uses a FIFO queue.\n\n"
        "Guarantees the shortest path in an unweighted graph. Great for "
        "level-order traversals."
    ),
    "DFS": (
        "Depth-First Search",
        "O(V + E) time  ·  O(V) space",
        "Explores as far as possible down each branch before backtracking. "
        "Uses a stack (or recursion).\n\n"
        "Useful for cycle detection, topological sort, and maze solving."
    ),
    "Dijkstra": (
        "Dijkstra's Algorithm",
        "O((V + E) log V) time  ·  O(V) space",
        "Finds shortest paths from a source to all other nodes in a "
        "weighted graph with non-negative weights.\n\n"
        "Uses a min-heap priority queue. The backbone of GPS navigation "
        "and network routing."
    ),
    "A*": (
        "A* Pathfinding",
        "O(E) time (heuristic-dependent)  ·  O(V) space",
        "An informed search that uses a heuristic h(n) to guide exploration "
        "toward the goal. f(n) = g(n) + h(n), where g is cost so far and "
        "h is estimated remaining cost.\n\n"
        "With an admissible heuristic, always finds the optimal path while "
        "exploring far fewer nodes than Dijkstra."
    ),
}
