"""Qt (PySide6) desktop launcher for the AutoML dashboard."""

from __future__ import annotations

import sys
import threading
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication
from werkzeug.serving import make_server

from .dashboard import create_app
from .engine import AutoMLResult


def run_dashboard_qt(
    result: Optional[AutoMLResult] = None,
    *,
    host: str = "127.0.0.1",
    port: int = 5000,
    width: int = 1400,
    height: int = 900,
    title: str = "AutoML Decision Hub",
) -> None:
    """Launch the dashboard in a native Qt window using the existing HTML app."""

    app = QApplication.instance() or QApplication(sys.argv)

    flask_app = create_app(result)
    server = make_server(host, port, flask_app)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    view = QWebEngineView()
    view.setWindowTitle(title)
    view.resize(width, height)
    url = QUrl(f"http://{host}:{port}")
    view.load(url)
    view.show()

    def _shutdown_server() -> None:
        server.shutdown()

    app.aboutToQuit.connect(_shutdown_server)

    # Start the Qt event loop (no-op if already running)
    app.exec()
