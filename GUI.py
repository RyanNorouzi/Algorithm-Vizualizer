"""
gui.py - Main Application Window
==================================
The top-level QMainWindow that wires together:
  - A QTabWidget with three tabs: Sort / Graph / Pathfinding
  - Left sidebar with algorithm selector, speed slider, and info panel
  - A QTimer that drives the animation by calling next() on step generators
  - All three canvas widgets from visualizer.py

Architecture:
    MainWindow
    ├── QTabWidget
    │   ├── Tab 0: SortTab   (SortCanvas + controls)
    │   ├── Tab 1: GraphTab  (GraphCanvas + controls)
    │   └── Tab 2: GridTab   (GridCanvas + controls)
    └── Shared QTimer → advances current generator each tick
"""

import random
from typing import Optional

from PyQt6.QtCore    import QTimer, Qt, pyqtSlot
from PyQt6.QtGui     import QColor, QFont, QPalette, QIcon
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy,
    QSlider, QSplitter, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget, QSpinBox, QCheckBox,
)

from utils import (
    ALGORITHM_DESCRIPTIONS, Colors, DEFAULT_DELAY, MAX_DELAY_MS, MIN_DELAY_MS,
    generate_random_array,
)
from sorting_algorithms  import bubble_sort, heap_sort, merge_sort, quick_sort
from graph_algorithms    import bfs as graph_bfs, dfs as graph_dfs, dijkstra, make_sample_graph
from pathfinding         import astar, grid_bfs
from visualizer          import GraphCanvas, GridCanvas, SortCanvas


# ── Style helpers ──────────────────────────────────────────────────────────

def _btn(text: str, color: str = Colors.ACCENT, min_w: int = 90) -> QPushButton:
    btn = QPushButton(text)
    btn.setMinimumWidth(min_w)
    btn.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {color};
            color: {Colors.BG_DARK};
            border: none;
            border-radius: 6px;
            padding: 7px 14px;
        }}
        QPushButton:hover  {{ background: #5aa8ff; }}
        QPushButton:pressed{{ background: #2a6abf; }}
        QPushButton:disabled{{ background: {Colors.BORDER}; color: {Colors.TEXT_MUTED}; }}
    """)
    return btn


def _label(text: str, muted: bool = False, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    color = Colors.TEXT_MUTED if muted else Colors.TEXT_PRIMARY
    weight = "bold" if bold else "normal"
    lbl.setStyleSheet(f"color: {color}; font-family: Monospace; font-weight: {weight};")
    return lbl


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"background: {Colors.BORDER}; max-height: 1px;")
    return line


GLOBAL_STYLE = f"""
QWidget {{
    background: {Colors.BG_DARK};
    color: {Colors.TEXT_PRIMARY};
    font-family: 'JetBrains Mono', 'Fira Code', Monospace;
    font-size: 11px;
}}
QTabWidget::pane {{
    border: 1px solid {Colors.BORDER};
    border-radius: 6px;
    background: {Colors.BG_PANEL};
}}
QTabBar::tab {{
    background: {Colors.BG_PANEL};
    color: {Colors.TEXT_MUTED};
    border: 1px solid {Colors.BORDER};
    padding: 8px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-family: Monospace;
    font-weight: bold;
}}
QTabBar::tab:selected {{
    background: {Colors.BG_WIDGET};
    color: {Colors.TEXT_PRIMARY};
    border-bottom-color: {Colors.BG_WIDGET};
}}
QTabBar::tab:hover {{ color: {Colors.TEXT_PRIMARY}; }}
QComboBox {{
    background: {Colors.BG_WIDGET};
    color: {Colors.TEXT_PRIMARY};
    border: 1px solid {Colors.BORDER};
    border-radius: 5px;
    padding: 5px 10px;
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background: {Colors.BG_PANEL};
    color: {Colors.TEXT_PRIMARY};
    selection-background-color: {Colors.ACCENT};
    selection-color: {Colors.BG_DARK};
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {Colors.SLIDER_GROOVE};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {Colors.ACCENT};
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
QSlider::sub-page:horizontal {{ background: {Colors.ACCENT}; border-radius: 3px; }}
QTextEdit {{
    background: {Colors.BG_WIDGET};
    color: {Colors.TEXT_MUTED};
    border: 1px solid {Colors.BORDER};
    border-radius: 6px;
    padding: 8px;
    font-size: 10px;
    line-height: 1.5;
}}
QScrollBar:vertical {{
    background: {Colors.BG_WIDGET};
    width: 8px;
}}
QScrollBar::handle:vertical {{ background: {Colors.BORDER}; border-radius: 4px; }}
QSpinBox {{
    background: {Colors.BG_WIDGET};
    color: {Colors.TEXT_PRIMARY};
    border: 1px solid {Colors.BORDER};
    border-radius: 5px;
    padding: 4px 8px;
}}
"""


# ══════════════════════════════════════════════════════════════════════════
#  Sorting Tab
# ══════════════════════════════════════════════════════════════════════════

class SortTab(QWidget):
    ALGORITHMS = ["Bubble Sort", "Merge Sort", "Quick Sort", "Heap Sort"]
    GENERATORS = {
        "Bubble Sort": bubble_sort,
        "Merge Sort":  merge_sort,
        "Quick Sort":  quick_sort,
        "Heap Sort":   heap_sort,
    }

    def __init__(self, timer: QTimer, parent=None):
        super().__init__(parent)
        self._timer     = timer
        self._gen       = None
        self._running   = False
        self._arr_size  = 50

        self._build_ui()
        self._new_array()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ── Canvas ──────────────────────────────────────────────────────
        self.canvas = SortCanvas()
        root.addWidget(self.canvas, stretch=4)

        # ── Sidebar ─────────────────────────────────────────────────────
        side = QVBoxLayout()
        side.setSpacing(10)

        side.addWidget(_label("ALGORITHM", muted=True))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(self.ALGORITHMS)
        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)
        side.addWidget(self.algo_combo)

        side.addWidget(_divider())

        side.addWidget(_label("ARRAY SIZE", muted=True))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(10, 120)
        self.size_spin.setValue(self._arr_size)
        self.size_spin.valueChanged.connect(self._on_size_changed)
        side.addWidget(self.size_spin)

        side.addWidget(_divider())

        side.addWidget(_label("SPEED", muted=True))
        self.speed_slider = _make_speed_slider()
        side.addWidget(self.speed_slider)
        speed_row = QHBoxLayout()
        speed_row.addWidget(_label("Slow", muted=True))
        speed_row.addStretch()
        speed_row.addWidget(_label("Fast", muted=True))
        side.addLayout(speed_row)

        side.addWidget(_divider())

        # Control buttons
        btn_row1 = QHBoxLayout()
        self.start_btn = _btn("▶  Start", Colors.ACCENT)
        self.pause_btn = _btn("⏸  Pause", "#F0A030")
        self.pause_btn.setEnabled(False)
        btn_row1.addWidget(self.start_btn)
        btn_row1.addWidget(self.pause_btn)
        side.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.reset_btn = _btn("↺  Reset",  "#7D8590")
        self.new_btn   = _btn("⟳  New",    "#7D8590")
        btn_row2.addWidget(self.reset_btn)
        btn_row2.addWidget(self.new_btn)
        side.addLayout(btn_row2)

        side.addWidget(_divider())

        # Info text
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        side.addWidget(self.info_text)

        side.addStretch()

        sidebar_w = QWidget()
        sidebar_w.setMaximumWidth(240)
        sidebar_w.setLayout(side)
        root.addWidget(sidebar_w, stretch=1)

        # Wire buttons
        self.start_btn.clicked.connect(self._start)
        self.pause_btn.clicked.connect(self._pause)
        self.reset_btn.clicked.connect(self._reset)
        self.new_btn.clicked.connect(self._new_array)
        self.speed_slider.valueChanged.connect(self._on_speed)

        self._update_info()

    # ── Animation step ───────────────────────────────────────────────────

    def tick(self):
        if not self._gen:
            return
        try:
            step = next(self._gen)
            self.canvas.apply_step(step)
        except StopIteration:
            self._gen = None
            self._running = False
            self._update_buttons()

    # ── Button handlers ──────────────────────────────────────────────────

    def _start(self):
        if not self._running:
            algo = self.algo_combo.currentText()
            arr  = self.canvas._array or generate_random_array(self._arr_size)
            self._gen    = self.GENERATORS[algo](arr)
            self._running = True
            self._update_buttons()

    def _pause(self):
        self._running = not self._running
        self.pause_btn.setText("▶  Resume" if not self._running else "⏸  Pause")

    def _reset(self):
        self._gen     = None
        self._running = False
        self.canvas.set_array(self.canvas._array[:] if self.canvas._array else
                              generate_random_array(self._arr_size))
        self._update_buttons()

    def _new_array(self):
        self._gen     = None
        self._running = False
        arr = generate_random_array(self._arr_size)
        self.canvas.set_array(arr)
        self._update_buttons()

    def _on_algo_changed(self, _):
        self._reset()
        self._update_info()

    def _on_size_changed(self, val):
        self._arr_size = val
        self._new_array()

    def _on_speed(self, val):
        # Slider 1 (slow) → max delay; slider 100 (fast) → min delay
        delay = int(MAX_DELAY_MS - (val / 100) * (MAX_DELAY_MS - MIN_DELAY_MS))
        self._timer.setInterval(delay)

    def _update_buttons(self):
        self.start_btn.setEnabled(not self._running)
        self.pause_btn.setEnabled(self._gen is not None)
        self.pause_btn.setText("⏸  Pause")

    def _update_info(self):
        algo = self.algo_combo.currentText()
        if algo in ALGORITHM_DESCRIPTIONS:
            name, complexity, desc = ALGORITHM_DESCRIPTIONS[algo]
            self.info_text.setHtml(
                f"<b style='color:{Colors.ACCENT}'>{name}</b><br>"
                f"<span style='color:{Colors.BAR_SORTED}'>{complexity}</span><br><br>"
                f"<span style='color:{Colors.TEXT_MUTED}'>{desc.replace(chr(10),'<br>')}</span>"
            )

    def is_running(self) -> bool:
        return self._running


# ══════════════════════════════════════════════════════════════════════════
#  Graph Tab
# ══════════════════════════════════════════════════════════════════════════

class GraphTab(QWidget):
    ALGORITHMS = ["BFS", "DFS", "Dijkstra"]

    def __init__(self, timer: QTimer, parent=None):
        super().__init__(parent)
        self._timer   = timer
        self._gen     = None
        self._running = False
        self._nodes, self._edges = make_sample_graph()

        self._build_ui()
        self.canvas.set_graph(self._nodes, self._edges)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        self.canvas = GraphCanvas()
        root.addWidget(self.canvas, stretch=4)

        side = QVBoxLayout()
        side.setSpacing(10)

        side.addWidget(_label("ALGORITHM", muted=True))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(self.ALGORITHMS)
        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)
        side.addWidget(self.algo_combo)

        side.addWidget(_label("START NODE", muted=True))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, len(self._nodes) - 1)
        self.start_spin.setValue(0)
        side.addWidget(self.start_spin)

        side.addWidget(_divider())

        side.addWidget(_label("SPEED", muted=True))
        self.speed_slider = _make_speed_slider()
        self.speed_slider.setValue(60)
        side.addWidget(self.speed_slider)
        sp_row = QHBoxLayout()
        sp_row.addWidget(_label("Slow", muted=True))
        sp_row.addStretch()
        sp_row.addWidget(_label("Fast", muted=True))
        side.addLayout(sp_row)

        side.addWidget(_divider())

        btn_row1 = QHBoxLayout()
        self.start_btn = _btn("▶  Start", Colors.ACCENT)
        self.pause_btn = _btn("⏸  Pause", "#F0A030")
        self.pause_btn.setEnabled(False)
        btn_row1.addWidget(self.start_btn)
        btn_row1.addWidget(self.pause_btn)
        side.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.reset_btn  = _btn("↺  Reset", "#7D8590")
        self.new_graph_btn = _btn("⟳  New",  "#7D8590")
        btn_row2.addWidget(self.reset_btn)
        btn_row2.addWidget(self.new_graph_btn)
        side.addLayout(btn_row2)

        side.addWidget(_divider())

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        side.addWidget(self.info_text)

        side.addStretch()

        sidebar_w = QWidget()
        sidebar_w.setMaximumWidth(240)
        sidebar_w.setLayout(side)
        root.addWidget(sidebar_w)

        self.start_btn.clicked.connect(self._start)
        self.pause_btn.clicked.connect(self._pause)
        self.reset_btn.clicked.connect(self._reset)
        self.new_graph_btn.clicked.connect(self._new_graph)
        self.speed_slider.valueChanged.connect(self._on_speed)
        self._update_info()

    def tick(self):
        if not self._gen:
            return
        try:
            step = next(self._gen)
            self.canvas.apply_step(step)
        except StopIteration:
            self._gen     = None
            self._running = False
            self._update_buttons()

    def _start(self):
        if not self._running:
            algo  = self.algo_combo.currentText()
            start = self.start_spin.value()
            gens  = {"BFS": graph_bfs, "DFS": graph_dfs, "Dijkstra": dijkstra}
            self._gen     = gens[algo](self._nodes, self._edges, start)
            self._running = True
            self._update_buttons()

    def _pause(self):
        self._running = not self._running
        self.pause_btn.setText("▶  Resume" if not self._running else "⏸  Pause")

    def _reset(self):
        self._gen     = None
        self._running = False
        self.canvas.set_graph(self._nodes, self._edges)
        self._update_buttons()

    def _new_graph(self):
        import random as _r
        _r.seed()
        self._nodes, self._edges = make_sample_graph()
        self.start_spin.setMaximum(len(self._nodes) - 1)
        self._reset()

    def _on_algo_changed(self, _):
        self._reset()
        self._update_info()

    def _on_speed(self, val):
        delay = int(MAX_DELAY_MS - (val / 100) * (MAX_DELAY_MS - MIN_DELAY_MS))
        self._timer.setInterval(delay)

    def _update_buttons(self):
        self.start_btn.setEnabled(not self._running)
        self.pause_btn.setEnabled(self._gen is not None)
        self.pause_btn.setText("⏸  Pause")

    def _update_info(self):
        algo = self.algo_combo.currentText()
        if algo in ALGORITHM_DESCRIPTIONS:
            name, complexity, desc = ALGORITHM_DESCRIPTIONS[algo]
            self.info_text.setHtml(
                f"<b style='color:{Colors.ACCENT}'>{name}</b><br>"
                f"<span style='color:{Colors.BAR_SORTED}'>{complexity}</span><br><br>"
                f"<span style='color:{Colors.TEXT_MUTED}'>{desc.replace(chr(10),'<br>')}</span>"
            )

    def is_running(self) -> bool:
        return self._running


# ══════════════════════════════════════════════════════════════════════════
#  Pathfinding / Grid Tab
# ══════════════════════════════════════════════════════════════════════════

class GridTab(QWidget):
    ALGORITHMS = ["A*", "BFS"]

    def __init__(self, timer: QTimer, parent=None):
        super().__init__(parent)
        self._timer   = timer
        self._gen     = None
        self._running = False
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        self.canvas = GridCanvas(rows=22, cols=44)
        root.addWidget(self.canvas, stretch=4)

        side = QVBoxLayout()
        side.setSpacing(10)

        side.addWidget(_label("ALGORITHM", muted=True))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(self.ALGORITHMS)
        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)
        side.addWidget(self.algo_combo)

        side.addWidget(_divider())

        side.addWidget(_label("SPEED", muted=True))
        self.speed_slider = _make_speed_slider()
        self.speed_slider.setValue(80)
        side.addWidget(self.speed_slider)
        sp_row = QHBoxLayout()
        sp_row.addWidget(_label("Slow", muted=True))
        sp_row.addStretch()
        sp_row.addWidget(_label("Fast", muted=True))
        side.addLayout(sp_row)

        side.addWidget(_divider())

        # Legend
        legend_title = _label("LEGEND", muted=True)
        side.addWidget(legend_title)
        for color, text in [
            (Colors.NODE_START,    "Start (S)"),
            (Colors.NODE_END,      "End (E)"),
            (Colors.NODE_WALL,     "Wall"),
            (Colors.NODE_VISITED,  "Visited"),
            (Colors.NODE_FRONTIER, "Frontier"),
            (Colors.NODE_PATH,     "Path"),
        ]:
            row = QHBoxLayout()
            dot = QLabel("■")
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            row.addWidget(dot)
            row.addWidget(_label(text, muted=True))
            row.addStretch()
            side.addLayout(row)

        side.addWidget(_divider())

        btn_row1 = QHBoxLayout()
        self.start_btn = _btn("▶  Start", Colors.ACCENT)
        self.pause_btn = _btn("⏸  Pause", "#F0A030")
        self.pause_btn.setEnabled(False)
        btn_row1.addWidget(self.start_btn)
        btn_row1.addWidget(self.pause_btn)
        side.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.clear_walls_btn = _btn("↺ Clear", "#7D8590")
        self.maze_btn        = _btn("⊞ Maze",  "#7D8590")
        btn_row2.addWidget(self.clear_walls_btn)
        btn_row2.addWidget(self.maze_btn)
        side.addLayout(btn_row2)

        side.addWidget(_divider())
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(160)
        side.addWidget(self.info_text)

        side.addStretch()

        sidebar_w = QWidget()
        sidebar_w.setMaximumWidth(240)
        sidebar_w.setLayout(side)
        root.addWidget(sidebar_w)

        self.start_btn.clicked.connect(self._start)
        self.pause_btn.clicked.connect(self._pause)
        self.clear_walls_btn.clicked.connect(self._clear)
        self.maze_btn.clicked.connect(self._random_maze)
        self.speed_slider.valueChanged.connect(self._on_speed)
        self._update_info()

    def tick(self):
        if not self._gen:
            return
        try:
            step = next(self._gen)
            self.canvas.apply_step(step)
        except StopIteration:
            self._gen     = None
            self._running = False
            self._update_buttons()

    def _start(self):
        if not self._running:
            self.canvas.clear_search()
            grid  = self.canvas.get_grid()
            start = self.canvas._start
            end   = self.canvas._end
            algo  = self.algo_combo.currentText()
            if algo == "A*":
                self._gen = astar(grid, start, end)
            else:
                self._gen = grid_bfs(grid, start, end)
            self._running = True
            self._update_buttons()

    def _pause(self):
        self._running = not self._running
        self.pause_btn.setText("▶  Resume" if not self._running else "⏸  Pause")

    def _clear(self):
        self._gen     = None
        self._running = False
        self.canvas.reset()
        self._update_buttons()

    def _random_maze(self):
        """Generate a random maze using randomized wall placement (~35% density)."""
        import random
        self._clear()
        rows, cols = self.canvas.rows, self.canvas.cols
        from utils import CellState
        for r in range(rows):
            for c in range(cols):
                cell = (r, c)
                if cell not in (self.canvas._start, self.canvas._end):
                    if random.random() < 0.30:
                        self.canvas._grid[r][c] = CellState.WALL
        self.canvas.update()

    def _on_algo_changed(self, _):
        self._gen     = None
        self._running = False
        self.canvas.clear_search()
        self._update_buttons()
        self._update_info()

    def _on_speed(self, val):
        delay = int(MAX_DELAY_MS - (val / 100) * (MAX_DELAY_MS - MIN_DELAY_MS))
        self._timer.setInterval(delay)

    def _update_buttons(self):
        self.start_btn.setEnabled(not self._running)
        self.pause_btn.setEnabled(self._gen is not None)
        self.pause_btn.setText("⏸  Pause")

    def _update_info(self):
        algo = self.algo_combo.currentText()
        if algo in ALGORITHM_DESCRIPTIONS:
            name, complexity, desc = ALGORITHM_DESCRIPTIONS[algo]
            self.info_text.setHtml(
                f"<b style='color:{Colors.ACCENT}'>{name}</b><br>"
                f"<span style='color:{Colors.BAR_SORTED}'>{complexity}</span><br><br>"
                f"<span style='color:{Colors.TEXT_MUTED}'>{desc.replace(chr(10),'<br>')}</span>"
            )

    def is_running(self) -> bool:
        return self._running


# ══════════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Algorithm Visualizer")
        self.setMinimumSize(1000, 680)
        self.setStyleSheet(GLOBAL_STYLE)

        # Single shared timer drives all animation
        self._timer = QTimer(self)
        self._timer.setInterval(DEFAULT_DELAY)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._timer.start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title bar
        title_row = QHBoxLayout()
        title_lbl = QLabel("⚙  Algorithm Visualizer")
        title_lbl.setFont(QFont("Monospace", 16, QFont.Weight.Bold))
        title_lbl.setStyleSheet(f"color: {Colors.ACCENT};")
        subtitle = QLabel("Interactive visualizations of sorting, graph, and pathfinding algorithms")
        subtitle.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        title_row.addWidget(title_lbl)
        title_row.addSpacing(16)
        title_row.addWidget(subtitle)
        title_row.addStretch()
        layout.addLayout(title_row)

        layout.addWidget(_divider())

        # Tabs
        self.tabs = QTabWidget()
        self.sort_tab  = SortTab(self._timer)
        self.graph_tab = GraphTab(self._timer)
        self.grid_tab  = GridTab(self._timer)

        self.tabs.addTab(self.sort_tab,  "📊  Sorting")
        self.tabs.addTab(self.graph_tab, "🕸  Graph Search")
        self.tabs.addTab(self.grid_tab,  "🗺  Pathfinding")

        layout.addWidget(self.tabs)

    def _tick(self):
        idx = self.tabs.currentIndex()
        if   idx == 0: tab = self.sort_tab
        elif idx == 1: tab = self.graph_tab
        else:          tab = self.grid_tab

        if tab.is_running():
            tab.tick()


# ── Speed slider factory ───────────────────────────────────────────────────

def _make_speed_slider(default: int = 40) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(1, 100)
    s.setValue(default)
    return s
