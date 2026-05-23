from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap, QRadialGradient


_LOGO_CACHE: QPixmap | None = None


def draw_cvlab_background(widget) -> None:
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    width = max(1, widget.width())
    height = max(1, widget.height())
    rect = QRectF(0, 0, width, height)

    gradient = QLinearGradient(0, 0, width, height)
    gradient.setColorAt(0.0, QColor("#03060d"))
    gradient.setColorAt(0.45, QColor("#06121a"))
    gradient.setColorAt(1.0, QColor("#090812"))
    painter.fillRect(rect, gradient)

    painter.setPen(Qt.PenStyle.NoPen)
    for center, radius, color in (
        (QPointF(width * 0.16, height * 0.22), width * 0.26, QColor(30, 217, 192, 42)),
        (QPointF(width * 0.78, height * 0.18), width * 0.22, QColor(255, 209, 102, 30)),
        (QPointF(width * 0.70, height * 0.84), width * 0.28, QColor(255, 122, 168, 28)),
    ):
        glow = QRadialGradient(center, radius)
        glow.setColorAt(0.0, color)
        glow.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
        painter.setBrush(glow)
        painter.drawEllipse(center, radius, radius)

    logo = _logo_pixmap()
    if not logo.isNull():
        _draw_logo_field(painter, logo, width, height)
        _draw_brand_badge(painter, logo, width, height)

    painter.end()


def _draw_logo_field(painter: QPainter, logo: QPixmap, width: int, height: int) -> None:
    sizes = (72, 104, 138)
    positions = (
        (0.05, 0.10, 1),
        (0.30, 0.18, 0),
        (0.58, 0.09, 2),
        (0.86, 0.22, 0),
        (0.14, 0.58, 2),
        (0.38, 0.76, 0),
        (0.63, 0.58, 1),
        (0.88, 0.72, 2),
    )
    for index, (x_ratio, y_ratio, size_index) in enumerate(positions):
        size = sizes[size_index]
        x = width * x_ratio - size / 2
        y = height * y_ratio - size / 2

        glow_color = QColor("#7dffe8" if index % 2 == 0 else "#ffd166")
        glow_color.setAlpha(28)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow_color)
        painter.drawEllipse(QRectF(x - 18, y - 18, size + 36, size + 36))

        painter.setOpacity(0.12 if size < 120 else 0.09)
        painter.drawPixmap(QRectF(x, y, size, size), logo, QRectF(0, 0, logo.width(), logo.height()))
        painter.setOpacity(1.0)


def _draw_brand_badge(painter: QPainter, logo: QPixmap, width: int, height: int) -> None:
    logo_size = 54
    margin = 34
    x = width - margin - 190
    y = margin - 4

    badge_rect = QRectF(x - 14, y - 10, 210, 78)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(3, 8, 16, 175))
    painter.drawRoundedRect(badge_rect, 8, 8)

    glow = QRadialGradient(QPointF(x + logo_size / 2, y + logo_size / 2), 62)
    glow.setColorAt(0.0, QColor(125, 255, 232, 80))
    glow.setColorAt(1.0, QColor(125, 255, 232, 0))
    painter.setBrush(glow)
    painter.drawEllipse(QRectF(x - 28, y - 28, 110, 110))

    painter.setOpacity(0.96)
    painter.drawPixmap(
        QRectF(x, y, logo_size, logo_size),
        logo,
        QRectF(0, 0, logo.width(), logo.height()),
    )
    painter.setOpacity(1.0)

    painter.setFont(QFont("Menlo", 23, QFont.Weight.Black))
    text_x = x + logo_size + 12
    text_y = y + 39
    for alpha, offset in ((70, 2), (42, 5)):
        painter.setPen(QColor(125, 255, 232, alpha))
        painter.drawText(QPointF(text_x + offset, text_y + offset), "CVLAB")
    painter.setPen(QColor("#f1fffb"))
    painter.drawText(QPointF(text_x, text_y), "CVLAB")


def _logo_pixmap() -> QPixmap:
    global _LOGO_CACHE
    if _LOGO_CACHE is None:
        project_root = Path(__file__).resolve().parents[1]
        _LOGO_CACHE = QPixmap(str(project_root / "cvlablogo.png"))
    return _LOGO_CACHE
