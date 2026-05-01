"""
ui/overlay.py
Clutch.ai — PyQt5 Transparent Hint Overlay

Key properties:
  - Frameless, transparent, always-on-top
  - NEVER steals keyboard focus
  - On macOS: NSWindowSharingNone → invisible to Zoom/Meet screen capture
    while remaining visible to the local user
  - Auto-hides after HINT_DISPLAY_SECONDS

Usage (from pipeline.py):
    app = QApplication(sys.argv)
    overlay = HintOverlay()
    # From any thread:
    overlay.hint_signal.emit("- Hint 1\\n- Hint 2\\n- Hint 3")
"""

import os
import sys
import ctypes
import ctypes.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HINT_DISPLAY_MS = int(os.getenv("HINT_DISPLAY_SECONDS", "12")) * 1000

WINDOW_WIDTH    = 420
EDGE_INSET      = 20
BORDER_RADIUS   = 16

# Colours
BG_COLOR        = QColor(10, 15, 28, 230)     # deep navy, ~90% opaque
BORDER_COLOR    = QColor(255, 255, 255, 45)
ACCENT_COLOR    = QColor(0, 229, 255)          # cyan
ACCENT_HEX      = "#00E5FF"
TEXT_HEX        = "#F1F5F9"
DIM_HEX         = "#64748B"


# ---------------------------------------------------------------------------
# macOS: exclude window from screen-share capture
# ---------------------------------------------------------------------------

def _macos_hide_from_capture(win_id: int) -> None:
    """
    [NSWindow setSharingType: NSWindowSharingNone]
    Makes the window invisible to Zoom/Meet/QuickTime screen capture
    while staying fully visible on the local display.
    Also raises the window level above meeting software.
    """
    if sys.platform != "darwin":
        return
    try:
        libobjc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        libobjc.sel_registerName.restype = ctypes.c_void_p
        libobjc.objc_msgSend.restype     = ctypes.c_void_p
        libobjc.objc_msgSend.argtypes    = [ctypes.c_void_p, ctypes.c_void_p]

        view   = ctypes.c_void_p(win_id)
        ns_win = libobjc.objc_msgSend(
            view, libobjc.sel_registerName(b"window")
        )
        if not ns_win:
            print("[UI] macOS: NSWindow not found — skipping")
            return

        _ulong = ctypes.cast(
            libobjc.objc_msgSend,
            ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong),
        )
        _long = ctypes.cast(
            libobjc.objc_msgSend,
            ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong),
        )

        # NSWindowSharingNone = 0 → excluded from Zoom/Meet screen capture
        _ulong(ns_win, libobjc.sel_registerName(b"setSharingType:"), 0)

        # NSStatusWindowLevel = 25 → floats above meeting software chrome
        _long(ns_win, libobjc.sel_registerName(b"setLevel:"), 25)

        # NSWindowCollectionBehaviorCanJoinAllSpaces = 1<<3
        # → appears on every Mission Control Space/desktop
        _ulong(ns_win, libobjc.sel_registerName(b"setCollectionBehavior:"), 1 << 3)

        # Force to front in the current Space immediately
        libobjc.objc_msgSend(ns_win, libobjc.sel_registerName(b"orderFrontRegardless"))

        print("[UI] macOS: NSWindowSharingNone + AllSpaces + orderFront applied ✓")
    except Exception as exc:
        print(f"[UI] macOS screen-share invisibility failed: {exc}")


# ---------------------------------------------------------------------------
# Overlay widget
# ---------------------------------------------------------------------------

class HintOverlay(QWidget):
    """Thread-safe hint overlay. Emit hint_signal(str) from any thread."""

    hint_signal   = pyqtSignal(str)   # final complete hint → formatted display
    stream_signal = pyqtSignal(str)   # partial text during streaming → plain display

    def __init__(self) -> None:
        super().__init__()
        self._macos_applied = False
        self._streaming = False
        self._setup_window()
        self._setup_ui()
        self._setup_timer()
        self.hint_signal.connect(self.show_hint)
        self.stream_signal.connect(self._on_stream_chunk)

    # ------------------------------------------------------------------
    # Window flags
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.FramelessWindowHint       |
            Qt.WindowStaysOnTopHint      |
            Qt.Tool                      |
            Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFixedWidth(WINDOW_WIDTH)
        self._reposition()

    def _reposition(self) -> None:
        screens = QApplication.screens()

        # OVERLAY_SCREEN env var: integer index to pin to a specific screen.
        # Leave unset (or "auto") to follow the cursor.
        screen_idx = os.getenv("OVERLAY_SCREEN", "auto").strip()
        if screen_idx.isdigit() and int(screen_idx) < len(screens):
            screen_obj = screens[int(screen_idx)]
        else:
            from PyQt5.QtGui import QCursor
            screen_obj = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()

        screen = screen_obj.availableGeometry()
        x = screen.right() - WINDOW_WIDTH - EDGE_INSET
        y = screen.top() + EDGE_INSET
        self.move(x, y)

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(8)

        # Header
        header = QHBoxLayout()
        header.setSpacing(6)

        dot = QLabel("●")
        dot.setStyleSheet(f"color: {ACCENT_HEX}; font-size: 9px;")
        header.addWidget(dot)

        brand = QLabel("CLUTCH.AI  ·  LIVE")
        brand.setStyleSheet(
            f"color: {ACCENT_HEX}; font-size: 10px; font-weight: 700; "
            "letter-spacing: 2px;"
        )
        header.addWidget(brand)
        header.addStretch()
        root.addLayout(header)

        # Hint text — RichText so we can bold the definition line
        self._hint_label = QLabel("")
        self._hint_label.setWordWrap(True)
        self._hint_label.setTextFormat(Qt.RichText)
        self._hint_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._hint_label.setStyleSheet(
            f"color: {TEXT_HEX}; font-size: 13px; line-height: 170%;"
        )
        f = QFont("Helvetica Neue")
        f.setPixelSize(13)
        self._hint_label.setFont(f)
        root.addWidget(self._hint_label)

        # Footer
        self._footer = QLabel(f"Auto-hides in {HINT_DISPLAY_MS // 1000}s")
        self._footer.setStyleSheet(f"color: {DIM_HEX}; font-size: 10px;")
        root.addWidget(self._footer)

        self.setLayout(root)

    # ------------------------------------------------------------------
    # Auto-hide timer
    # ------------------------------------------------------------------

    def _setup_timer(self) -> None:
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    # ------------------------------------------------------------------
    # Custom painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()

        # Background
        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, BORDER_RADIUS, BORDER_RADIUS)
        painter.setPen(Qt.NoPen)
        painter.setBrush(BG_COLOR)
        painter.drawPath(path)

        # Border rim
        painter.setPen(BORDER_COLOR)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

        # Cyan top accent bar (3 px)
        accent = QPainterPath()
        accent.addRoundedRect(0, 0, w, 3, 2, 2)
        painter.setPen(Qt.NoPen)
        painter.setBrush(ACCENT_COLOR)
        painter.drawPath(accent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _format_hint(text: str) -> str:
        """
        Convert plain-text hint to HTML for the overlay.
        Now simply escapes the conversational paragraph.
        """
        import html as _html
        lines = text.strip().splitlines()
        parts = [_html.escape(line) for line in lines if line.strip()]
        return "<br><br>".join(parts)

    # ------------------------------------------------------------------
    # Streaming support
    # ------------------------------------------------------------------

    def _on_stream_chunk(self, partial: str) -> None:
        """
        Called on every streaming token from the LLM.
        Shows the window immediately on first chunk; updates plain text
        as tokens arrive. The final show_hint() call switches to
        formatted HTML and starts the auto-hide timer.
        """
        self._streaming = True

        if not self.isVisible():
            self._reposition()
            self.setWindowOpacity(1.0)
            self.show()
            self.raise_()
            if not self._macos_applied:
                _macos_hide_from_capture(int(self.winId()))
                self._macos_applied = True
            self._footer.setText("Generating ...")

        # Plain text during streaming — fastest possible render
        self._hint_label.setTextFormat(Qt.PlainText)
        self._hint_label.setText(partial.strip())
        self.adjustSize()

    # ------------------------------------------------------------------
    # Final display (after streaming complete or direct call)
    # ------------------------------------------------------------------

    def show_hint(self, text: str) -> None:
        """
        Display the final formatted hint. Safe to call from any thread
        via hint_signal. Applies rich HTML formatting and starts the
        auto-hide timer.
        """
        self._streaming = False
        secs = HINT_DISPLAY_MS // 1000

        # Switch to rich-text for formatting
        self._hint_label.setTextFormat(Qt.RichText)
        self._hint_label.setText(self._format_hint(text))
        self._footer.setText(f"Auto-hides in {secs}s")

        self.adjustSize()
        self._reposition()

        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()

        if not self._macos_applied:
            _macos_hide_from_capture(int(self.winId()))
            self._macos_applied = True

        self._timer.stop()
        self._timer.start(HINT_DISPLAY_MS)
        print(f"[UI] Hint displayed ({secs}s)")

    def _on_timer(self) -> None:
        self.hide()
        self._streaming = False
        print("[UI] Hint hidden")


# ---------------------------------------------------------------------------
# Standalone test  —  python ui/overlay.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import threading
    import time

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    overlay = HintOverlay()

    # Show immediately so we can see it
    overlay.show_hint("⚡ Clutch.ai ready — speak a technical question ...")

    def _send_test():
        time.sleep(3)
        overlay.hint_signal.emit(
            "- BST: left < node < right, avg O(log n)\n"
            "- In-order traversal yields sorted sequence\n"
            "- Degenerate (sorted input) → O(n); prefer AVL or Red-Black"
        )

    threading.Thread(target=_send_test, daemon=True).start()

    QTimer.singleShot(20_000, app.quit)
    sys.exit(app.exec_())
