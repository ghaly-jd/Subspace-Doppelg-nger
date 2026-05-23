from __future__ import annotations


APP_STYLESHEET = """
QMainWindow {
    background: #05070d;
}

QWidget {
    background: transparent;
    color: #d8fff7;
    font-family: "Menlo", "Consolas", "Courier New", monospace;
    font-size: 15px;
}

QLabel {
    background: transparent;
}

QLabel#titleLabel {
    background: transparent;
    color: #7dffe8;
    font-size: 28px;
    font-weight: 700;
}

QLabel#subtitleLabel {
    background: transparent;
    color: #8aa4b8;
    font-size: 14px;
}

QLabel#panelTitle {
    background: transparent;
    color: #7dffe8;
    font-size: 22px;
    font-weight: 700;
}

QLabel#statusLine {
    background: transparent;
    color: #8aa4b8;
}

QLineEdit,
QComboBox {
    background: #071724;
    border: 1px solid #1fd9c0;
    color: #d8fff7;
    min-height: 34px;
    padding: 6px 10px;
}

QComboBox QAbstractItemView {
    background: #071724;
    border: 1px solid #1fd9c0;
    color: #d8fff7;
    selection-background-color: #0b2636;
}

QLabel#phasePromptTitle {
    background: #0b2636;
    border: 2px solid #7dffe8;
    color: #f1fffb;
    font-size: 34px;
    font-weight: 800;
    padding: 14px 18px;
}

QLabel#phasePromptBody {
    background: #071724;
    border-left: 2px solid #1fd9c0;
    border-right: 2px solid #1fd9c0;
    border-bottom: 2px solid #1fd9c0;
    color: #ffd166;
    font-size: 22px;
    font-weight: 700;
    padding: 10px 18px;
}

QPushButton {
    background: #071724;
    border: 1px solid #1fd9c0;
    color: #d8fff7;
    min-height: 42px;
    padding: 8px 16px;
    text-align: left;
}

QPushButton:hover {
    background: #0b2636;
    border-color: #7dffe8;
}

QPushButton:pressed {
    background: #10384d;
}

QFrame#terminalPanel {
    border: 1px solid rgba(125, 255, 232, 125);
    background: rgba(7, 16, 26, 224);
}
"""


def apply_theme(app) -> None:
    app.setStyleSheet(APP_STYLESHEET)
