"""
ui/overlay.py
Clutch.ai — PyQt5 Transparent Hint Overlay

A frameless, transparent, always-on-top PyQt5 window that displays 3-bullet
hint text. Auto-hides after 12 seconds. Never steals keyboard focus.
Thread-safe via Qt signal/slot mechanism.

Usage (from pipeline.py):
    app = QApplication(sys.argv)
    overlay = HintOverlay()
    overlay.hint_signal.connect(overlay.show_hint)
    # From worker thread:
    overlay.hint_signal.emit("- Hint 1\n- Hint 2\n- Hint 3")
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

# ---------------------------------------------------------------------------
# Constants (from .env or defaults)
# ---------------------------------------------------------------------------
HINT_DISPLAY_SECONDS = int(os.getenv("HINT_DISPLAY_SECONDS", "12")) * 1000  # ms

WINDOW_WIDTH  = 380
WINDOW_MAX_H  = 220
INSET_PX      = 20
BORDER_RADIUS = 12

BG_COLOR      = (15, 20, 30, int(0.88 * 255))   # rgba(15, 20, 30, 0.88)
ACCENT_COLOR  = "#3B82F6"                         # Blue top accent
TEXT_COLOR    = "#FFFFFF"
FONT_SIZE     = 13
FONT_FAMILY   = "Inter, Segoe UI, SF Pro Display, sans-serif"

LOGO_TEXT     = "⚡ clutch.ai"


class HintOverlay(QWidget):
    """
    Frameless, transparent, always-on-top overlay for displaying interview hints.

    Signals:
        hint_signal(str): Emit from any thread to display a hint safely.
    """

    hint_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._setup_window()
        self._setup_ui()
        self._setup_timer()

        # Connect signal to slot (thread-safe)
        self.hint_signal.connect(self.show_hint)

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """Configure window flags and attributes."""
        self.setWindowFlags(
            Qt.FramelessWindowHint      |   # No title bar or borders
            Qt.WindowStaysOnTopHint     |   # Always on top
            Qt.Tool                     |   # No taskbar entry
            Qt.WindowDoesNotAcceptFocus     # Never steals focus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setFixedWidth(WINDOW_WIDTH)

        # Position: bottom-right corner of primary screen
        self._reposition()

    def _reposition(self) -> None:
        """Move to bottom-right of the primary screen."""
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.right()  - WINDOW_WIDTH - INSET_PX
        y = screen.bottom() - WINDOW_MAX_H  - INSET_PX
        self.move(x, y)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the internal layout: logo label + hint label."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(6)

        # Logo / header
        self._logo_label = QLabel(LOGO_TEXT)
        self._logo_label.setStyleSheet(
            f"color: {ACCENT_COLOR}; font-size: 11px; font-weight: bold; "
            "letter-spacing: 1px;"
        )
        layout.addWidget(self._logo_label)

        # Hint text
        self._hint_label = QLabel("")
        self._hint_label.setWordWrap(True)
        self._hint_label.setTextFormat(Qt.PlainText)
        self._hint_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._hint_label.setStyleSheet(
            f"color: {TEXT_COLOR}; font-size: {FONT_SIZE}px; "
            "line-height: 1.6; padding-left: 2px;"
        )
        font = QFont()
        font.setPixelSize(FONT_SIZE)
        self._hint_label.setFont(font)
        layout.addWidget(self._hint_label)

        self.setLayout(layout)

    def _setup_timer(self) -> None:
        """Set up the auto-hide timer."""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timer)

    # ------------------------------------------------------------------
    # Custom painting — rounded background with top accent line
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        r, g, b, a = BG_COLOR
        painter.setBrush(QColor(r, g, b, a))
        painter.setPen(Qt.NoPen)

        # Rounded rect background
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), BORDER_RADIUS, BORDER_RADIUS)
        painter.drawPath(path)

        # Top accent line (3px, blue)
        painter.setBrush(QColor(ACCENT_COLOR))
        painter.drawRoundedRect(0, 0, self.width(), 3, 2, 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_hint(self, text: str) -> None:
        """
        Display a hint. Called from the main thread via signal/slot.

        Args:
            text: Multi-line string of dash-prefixed bullet points.
        """
        # Clean up the text — ensure each bullet is on its own line
        formatted = text.strip()
        self._hint_label.setText(formatted)

        # Resize to fit content
        self.adjustSize()
        self.setMaximumHeight(WINDOW_MAX_H)
        self._reposition()

        self.show()
        self.raise_()

        # Restart 12-second auto-hide timer
        self._timer.stop()
        self._timer.start(HINT_DISPLAY_SECONDS)

        print(f"[UI] Hint displayed ({HINT_DISPLAY_SECONDS // 1000}s)")

    def _on_timer(self) -> None:
        """Called when the display timer fires — hides the overlay."""
        self.hide()
        print("[UI] Hint hidden (timer expired)")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import threading
    import time

    app = QApplication(sys.argv)

    overlay = HintOverlay()
    overlay.show_hint("Clutch.ai ready...")

    def _send_test_hint():
        time.sleep(2)
        hint = (
            "- BST: left < node < right, O(log n) average\n"
            "- In-order traversal gives sorted output\n"
            "- Degenerate case: sorted input → O(n) — use AVL or Red-Black"
        )
        overlay.hint_signal.emit(hint)

    t = threading.Thread(target=_send_test_hint, daemon=True)
    t.start()

    sys.exit(app.exec_())
