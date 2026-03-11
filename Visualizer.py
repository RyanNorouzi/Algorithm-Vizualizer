"""
visualizer.py - Custom PyQt6 Canvas Widgets
=============================================
Three QPainter-based widgets, one per algorithm category:

  SortCanvas    — animated bar chart for sorting algorithms
  GraphCanvas   — node-and-edge graph for BFS / DFS / Dijkstra
  GridCanvas    — interactive 2-D grid for A* / BFS pathfinding

Each widget is a standalone QWidget with a draw() method and
a clear() method. The main GUI drives them via QTimer.
"""

import math
from typing import Dict, List, Optional, Set, Tuple

from PyQt6.QtCore    import Qt, QPoint, QRect, QRectF, QTimer
from PyQt6.QtGui     import (QColor, QFont, QPainter, QPainterPath,
                              QPen, QBrush, QFontMetrics, QLinearGradient)
from PyQt6.QtWidgets import QWidget, QSizePolicy

from utils import (BarState, CellState, Colors, GraphEdge, GraphNode,
                   PathStep, SortStep, bar_color, cell_color)


# ── Shared painter helpers ─────────────────────────────────────────────────

def qc(hex_color: str) -> QColor:
    """Convert hex string to QColor."""
    return QColor(hex_color)


# ══════════════════════════════════════════════════════════════════════════
#  Sort Canvas
# ══════════════════════════════════════════════════════════════════════════

class SortCanvas(QWidget):
    """
    Animated bar-chart visualizer for sorting algorithms.
    
    Usage:
        canvas.set_array([40, 10, 80, ...])   # set initial array
        canvas.apply_step(sort_step)           # apply one SortStep
        canvas.clear()                         # reset to defaults
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 380)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background: {Colors.BG_WIDGET}; border-radius: 8px;")

        self._array: List[int]      = []
        self._highlights: Dict      = {}
        self._message: str          = ""
        self._comparisons: int      = 0
        self._swaps: int            = 0

    # ── Public API ──────────────────────────────────────────────────────

    def set_array(self, arr: List[int]) -> None:
        self._array      = arr[:]
        self._highlights = {}
        self._message    = "Ready"
        self._comparisons = 0
        self._swaps       = 0
        self.update()

    def apply_step(self, step: SortStep) -> None:
        self._array      = step.array[:]
        self._highlights = step.highlights.copy()
        self._message    = step.message

        # Count stats from highlights
        for state in step.highlights.values():
            if state == BarState.COMPARE:
                self._comparisons += 1
            elif state == BarState.SWAP:
                self._swaps += 1
        self.update()

    def clear(self) -> None:
        self._array      = []
        self._highlights = {}
        self._message    = ""
        self._comparisons = 0
        self._swaps       = 0
        self.update()

    # ── Paint ───────────────────────────────────────────────────────────

    def paintEvent(self, _):
        if not self._array:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, qc(Colors.BG_WIDGET))

        n      = len(self._array)
        max_val = max(self._array) if self._array else 1

        # Reserve space for stats bar at top and message bar at bottom
        TOP_PAD = 40
        BOT_PAD = 36
        bar_area_h = h - TOP_PAD - BOT_PAD

        bar_w     = max(2, (w - 20) / n)
        gap       = max(1, bar_w * 0.12)
        bar_width = bar_w - gap

        for i, val in enumerate(self._array):
            state    = self._highlights.get(i, BarState.DEFAULT)
            color    = qc(bar_color(state))

            bar_h  = max(4, int(val / max_val * bar_area_h * 0.92))
            x      = int(10 + i * bar_w)
            y      = h - BOT_PAD - bar_h

            # Draw bar with subtle rounded top
            rect = QRectF(x, y, max(2, bar_width), bar_h)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 2, 2)

            # Draw value label only if bars are wide enough
            if bar_width >= 14 and n <= 60:
                painter.setPen(QPen(qc(Colors.TEXT_PRIMARY)))
                painter.setFont(QFont("Monospace", 7))
                painter.drawText(QRect(x, y - 16, int(bar_width), 14),
                                 Qt.AlignmentFlag.AlignHCenter, str(val))

        # Stats row (top)
        painter.setPen(QPen(qc(Colors.TEXT_MUTED)))
        painter.setFont(QFont("Monospace", 10))
        painter.drawText(QRect(10, 8, 300, 24), Qt.AlignmentFlag.AlignLeft,
                         f"Comparisons: {self._comparisons}   Swaps: {self._swaps}")
        painter.drawText(QRect(0, 8, w - 10, 24), Qt.AlignmentFlag.AlignRight,
                         f"n = {n}")

        # Message row (bottom)
        painter.setPen(QPen(qc(Colors.TEXT_PRIMARY)))
        painter.setFont(QFont("Monospace", 10))
        painter.drawText(QRect(10, h - BOT_PAD + 8, w - 20, 24),
                         Qt.AlignmentFlag.AlignLeft, self._message)
        painter.end()


# ══════════════════════════════════════════════════════════════════════════
#  Graph Canvas
# ══════════════════════════════════════════════════════════════════════════

class GraphCanvas(QWidget):
    """
    Node-and-edge canvas for BFS, DFS, and Dijkstra.
    
    Usage:
        canvas.set_graph(nodes, edges)
        canvas.apply_step(path_step)
        canvas.clear()
    """

    NODE_RADIUS = 22

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background: {Colors.BG_WIDGET}; border-radius: 8px;")

        self._nodes:      List[GraphNode] = []
        self._edges:      List[GraphEdge] = []
        self._visited:    Set[int]        = set()
        self._frontier:   Set[int]        = set()
        self._path:       List[int]       = []
        self._active_edge: tuple          = ()
        self._distances:  Dict[int, float] = {}
        self._message:    str             = ""

    # ── Public API ──────────────────────────────────────────────────────

    def set_graph(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> None:
        self._nodes    = nodes
        self._edges    = edges
        self._visited  = set()
        self._frontier = set()
        self._path     = []
        self._active_edge = ()
        self._distances = {}
        self._message  = "Select an algorithm and press Start"
        self.update()

    def apply_step(self, step: PathStep) -> None:
        self._visited    = set(step.visited)
        self._frontier   = set(step.frontier)
        self._path       = list(step.path)
        self._active_edge = step.active_edge
        self._distances  = dict(step.distances)
        self._message    = step.message
        self.update()

    def clear(self) -> None:
        self._visited  = set()
        self._frontier = set()
        self._path     = []
        self._active_edge = ()
        self._distances = {}
        self._message  = ""
        self.update()

    # ── Paint ───────────────────────────────────────────────────────────

    def paintEvent(self, _):
        if not self._nodes:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(0, 0, self.width(), self.height(), qc(Colors.BG_WIDGET))

        # Scale node positions to fit current widget size
        sx = self.width()  / 700
        sy = self.height() / 480

        def pos(node: GraphNode) -> QPoint:
            return QPoint(int(node.x * sx), int(node.y * sy))

        # Draw edges
        for edge in self._edges:
            if edge.src >= edge.dst:
                continue   # Draw each undirected edge once
            src_pos = pos(self._nodes[edge.src])
            dst_pos = pos(self._nodes[edge.dst])

            active = (self._active_edge == (edge.src, edge.dst) or
                      self._active_edge == (edge.dst, edge.src))
            color  = Colors.EDGE_ACTIVE if active else Colors.EDGE_DEFAULT
            width  = 3 if active else 1

            painter.setPen(QPen(qc(color), width))
            painter.drawLine(src_pos, dst_pos)

            # Weight label
            mid = QPoint((src_pos.x() + dst_pos.x()) // 2,
                         (src_pos.y() + dst_pos.y()) // 2)
            painter.setPen(QPen(qc(Colors.TEXT_MUTED)))
            painter.setFont(QFont("Monospace", 8))
            painter.drawText(mid, f"{edge.weight:.0f}")

        # Draw nodes
        R = self.NODE_RADIUS
        for node in self._nodes:
            nid = node.node_id
            p   = pos(node)

            # Choose fill color based on state
            if nid in self._path:
                fill = Colors.NODE_PATH
            elif nid in self._frontier:
                fill = Colors.NODE_FRONTIER
            elif nid in self._visited:
                fill = Colors.NODE_VISITED
            else:
                fill = Colors.NODE_DEFAULT

            # Glow ring for frontier
            if nid in self._frontier:
                painter.setPen(QPen(qc(Colors.NODE_FRONTIER), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(p, R + 5, R + 5)

            # Node circle
            painter.setBrush(QBrush(qc(fill)))
            painter.setPen(QPen(qc(Colors.BORDER), 2))
            painter.drawEllipse(p, R, R)

            # Node label
            painter.setPen(QPen(qc(Colors.TEXT_PRIMARY)))
            painter.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
            painter.drawText(QRect(p.x() - R, p.y() - R, 2 * R, 2 * R),
                             Qt.AlignmentFlag.AlignCenter, str(nid))

            # Distance label (Dijkstra)
            if nid in self._distances and self._distances[nid] != float("inf"):
                painter.setPen(QPen(qc(Colors.ACCENT)))
                painter.setFont(QFont("Monospace", 8))
                painter.drawText(p.x() - R, p.y() - R - 4,
                                 f"{self._distances[nid]:.1f}")

        # Message bar
        painter.setPen(QPen(qc(Colors.TEXT_MUTED)))
        painter.setFont(QFont("Monospace", 10))
        painter.drawText(QRect(10, self.height() - 28, self.width() - 20, 24),
                         Qt.AlignmentFlag.AlignLeft, self._message)
        painter.end()


# ══════════════════════════════════════════════════════════════════════════
#  Grid Canvas
# ══════════════════════════════════════════════════════════════════════════

class GridCanvas(QWidget):
    """
    Interactive 2-D grid for A* and BFS pathfinding.
    
    Left-click/drag  → place WALL cells
    Right-click      → erase cell (set to EMPTY)
    Start/End nodes are set via set_start() / set_end().
    
    Usage:
        canvas.reset()
        canvas.set_start((2, 2))
        canvas.set_end((18, 28))
        canvas.apply_step(path_step)
    """

    def __init__(self, rows: int = 22, cols: int = 42, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.setMinimumSize(600, 340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background: {Colors.BG_WIDGET}; border-radius: 8px;")
        self.setMouseTracking(True)

        self._grid: List[List[CellState]] = []
        self._start: Optional[Tuple[int, int]] = None
        self._end:   Optional[Tuple[int, int]] = None
        self._path:  List[Tuple[int, int]]     = []
        self._visited: Set[Tuple[int, int]]    = set()
        self._frontier: Set[Tuple[int, int]]   = set()
        self._message = ""
        self._drawing_walls = False
        self._erasing       = False

        self.reset()

    # ── Public API ──────────────────────────────────────────────────────

    def reset(self) -> None:
        self._grid   = [[CellState.EMPTY] * self.cols for _ in range(self.rows)]
        self._start  = (self.rows // 2, 3)
        self._end    = (self.rows // 2, self.cols - 4)
        self._path   = []
        self._visited  = set()
        self._frontier = set()
        self._message  = "Left-click to draw walls  ·  Right-click to erase"
        self.update()

    def set_start(self, pos: Tuple[int, int]) -> None:
        self._start = pos
        self.update()

    def set_end(self, pos: Tuple[int, int]) -> None:
        self._end = pos
        self.update()

    def get_grid(self):
        return [row[:] for row in self._grid]

    def apply_step(self, step: PathStep) -> None:
        self._visited  = set(step.visited)
        self._frontier = set(step.frontier)
        self._path     = list(step.path)
        self._message  = step.message
        self.update()

    def clear_search(self) -> None:
        self._visited  = set()
        self._frontier = set()
        self._path     = []
        self._message  = ""
        self.update()

    # ── Cell geometry ────────────────────────────────────────────────────

    def _cell_size(self) -> Tuple[float, float]:
        return self.width() / self.cols, self.height() / self.rows

    def _cell_at(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        cw, ch = self._cell_size()
        col = int(px / cw)
        row = int(py / ch)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    # ── Mouse interaction ────────────────────────────────────────────────

    def mousePressEvent(self, event):
        cell = self._cell_at(event.pos().x(), event.pos().y())
        if not cell:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing_walls = True
            self._toggle_wall(cell)
        elif event.button() == Qt.MouseButton.RightButton:
            self._erasing = True
            self._erase_cell(cell)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)):
            return
        cell = self._cell_at(event.pos().x(), event.pos().y())
        if not cell:
            return
        if self._drawing_walls:
            self._toggle_wall(cell)
        elif self._erasing:
            self._erase_cell(cell)

    def mouseReleaseEvent(self, _):
        self._drawing_walls = False
        self._erasing       = False

    def _toggle_wall(self, cell: Tuple[int, int]) -> None:
        r, c = cell
        if cell not in (self._start, self._end):
            self._grid[r][c] = CellState.WALL
            self.update()

    def _erase_cell(self, cell: Tuple[int, int]) -> None:
        r, c = cell
        if cell not in (self._start, self._end):
            self._grid[r][c] = CellState.EMPTY
            self.update()

    # ── Paint ────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        cw, ch = self._cell_size()

        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)
                state = self._grid[r][c]

                # Determine display color (search results override base state)
                if cell == self._start:
                    color = Colors.NODE_START
                elif cell == self._end:
                    color = Colors.NODE_END
                elif cell in self._path:
                    color = Colors.NODE_PATH
                elif cell in self._frontier:
                    color = Colors.NODE_FRONTIER
                elif cell in self._visited:
                    color = Colors.NODE_VISITED
                else:
                    color = cell_color(state)

                x = int(c * cw)
                y = int(r * ch)
                w = max(1, int(cw))
                h = max(1, int(ch))

                painter.fillRect(x, y, w, h, qc(color))

                # Grid lines
                painter.setPen(QPen(qc(Colors.BORDER), 0.5))
                painter.drawRect(x, y, w, h)

        # Start / End markers
        for pos, label, color in [(self._start, "S", Colors.NODE_START),
                                   (self._end,   "E", Colors.NODE_END)]:
            if pos:
                r, c = pos
                x = int(c * cw) + 1
                y = int(r * ch) + 1
                painter.setPen(QPen(qc(Colors.BG_DARK)))
                painter.setFont(QFont("Monospace", max(7, int(min(cw, ch) * 0.55)),
                                      QFont.Weight.Bold))
                painter.drawText(QRect(x, y, int(cw), int(ch)),
                                 Qt.AlignmentFlag.AlignCenter, label)

        # Message bar
        painter.setPen(QPen(qc(Colors.TEXT_MUTED)))
        painter.setFont(QFont("Monospace", 9))
        painter.drawText(QRect(6, self.height() - 20, self.width() - 12, 18),
                         Qt.AlignmentFlag.AlignLeft, self._message)
        painter.end()
