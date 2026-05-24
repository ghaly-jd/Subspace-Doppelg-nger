from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QMouseEvent, QPainter, QPen, QWheelEvent
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from shared.subspace import compare_subspaces
from ui.base_screen import BaseScreen

try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
except ImportError:
    QOpenGLWidget = QWidget


@dataclass(frozen=True)
class GalaxyNode:
    label: str
    position: np.ndarray
    similarity: float
    mean_angle_degrees: float
    projection_distance: float
    canonical_angles_degrees: tuple[float, ...]
    color: QColor
    is_query: bool = False


class SubspaceGalaxyCanvas(QOpenGLWidget):
    def __init__(self) -> None:
        super().__init__()
        self.nodes: list[GalaxyNode] = []
        self.status_text = "No subspace galaxy loaded."
        self.setMinimumSize(620, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self.rotation_x = -18.0
        self.rotation_y = 28.0
        self.camera_distance = 3.2
        self.last_mouse_pos: QPoint | None = None
        self.idle_rotation = True

        self.starfield = self._build_starfield()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def set_nodes(self, nodes: list[GalaxyNode], status_text: str) -> None:
        self.nodes = nodes
        self.status_text = status_text
        self.update()

    def reset_view(self) -> None:
        self.rotation_x = -18.0
        self.rotation_y = 28.0
        self.camera_distance = 3.2
        self.idle_rotation = True
        self.update()

    def zoom_in(self) -> None:
        self.camera_distance = max(1.35, self.camera_distance * 0.88)
        self.update()

    def zoom_out(self) -> None:
        self.camera_distance = min(7.0, self.camera_distance / 0.88)
        self.update()

    def initializeGL(self) -> None:
        self.setAutoFillBackground(False)

    def paintGL(self) -> None:
        self._paint_scene()

    def paintEvent(self, event) -> None:
        if QOpenGLWidget is QWidget:
            self._paint_scene()
            return
        super().paintEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.position().toPoint()
            self.idle_rotation = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.last_mouse_pos is None:
            return
        pos = event.position().toPoint()
        delta = pos - self.last_mouse_pos
        self.rotation_y += delta.x() * 0.45
        self.rotation_x += delta.y() * 0.45
        self.rotation_x = max(-85.0, min(85.0, self.rotation_x))
        self.last_mouse_pos = pos
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.last_mouse_pos = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.reset_view()

    def wheelEvent(self, event: QWheelEvent) -> None:
        angle_delta = event.angleDelta().y()
        pixel_delta = event.pixelDelta().y()
        steps = angle_delta / 120.0 if angle_delta else pixel_delta / 48.0
        if abs(steps) < 0.01:
            return
        self.camera_distance = max(
            1.35,
            min(7.0, self.camera_distance * math.exp(-steps * 0.09)),
        )
        self.update()
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_in()
            return
        if event.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.zoom_out()
            return
        super().keyPressEvent(event)

    def _tick(self) -> None:
        if self.idle_rotation and self.nodes:
            self.rotation_y += 0.08
            self.update()

    def _paint_scene(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_background(painter)
        if len(self.nodes) < 2:
            self._draw_empty_state(painter)
            painter.end()
            return

        projected = self._project_nodes()
        self._draw_distance_lines(painter, projected)
        self._draw_nodes(painter, projected)
        self._draw_overlay(painter)
        painter.end()

    def _draw_background(self, painter: QPainter) -> None:
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor("#02040b"))
        gradient.setColorAt(0.55, QColor("#061321"))
        gradient.setColorAt(1.0, QColor("#02040b"))
        painter.fillRect(self.rect(), gradient)

        painter.setPen(Qt.PenStyle.NoPen)
        for x, y, depth, size, alpha in self._project_starfield():
            painter.setBrush(QColor(125, 255, 232, alpha))
            painter.drawEllipse(QPointF(x, y), size / max(0.8, depth * 0.35), size / max(0.8, depth * 0.35))

    def _draw_empty_state(self, painter: QPainter) -> None:
        painter.setPen(QColor("#8aa4b8"))
        painter.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.status_text)

    def _draw_distance_lines(self, painter: QPainter, projected: list[tuple[GalaxyNode, QPointF, float]]) -> None:
        query = next((item for item in projected if item[0].is_query), None)
        if query is None:
            return
        _, query_point, _ = query
        for node, point, depth in projected:
            if node.is_query:
                continue
            alpha = int(45 + max(0.0, min(100.0, node.similarity)) * 1.25)
            color = QColor("#7dffe8")
            color.setAlpha(alpha)
            painter.setPen(QPen(color, max(1.0, 3.2 - depth * 0.8)))
            painter.drawLine(query_point, point)

    def _draw_nodes(self, painter: QPainter, projected: list[tuple[GalaxyNode, QPointF, float]]) -> None:
        for node, point, depth in sorted(projected, key=lambda item: item[2], reverse=True):
            radius = 14.0 if node.is_query else 7.0 + max(0.0, node.similarity) / 16.0
            radius *= max(0.72, min(1.35, 1.22 - depth * 0.12))
            glow = QColor(node.color)
            glow.setAlpha(52 if node.is_query else 34)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(point, radius * 2.4, radius * 2.4)

            color = QColor(node.color)
            color.setAlpha(235)
            painter.setBrush(color)
            painter.drawEllipse(point, radius, radius)

            painter.setPen(QPen(QColor("#eaffff"), 1))
            painter.setFont(QFont("Menlo", 10, QFont.Weight.Bold))
            label = node.label if node.is_query else f"{node.label} {node.similarity:.0f}%"
            painter.drawText(point + QPointF(radius + 8, -radius - 3), label)

    def _draw_overlay(self, painter: QPainter) -> None:
        top_match = next((node for node in self.nodes if not node.is_query), None)
        painter.setPen(QColor("#d8fff7"))
        painter.setFont(QFont("Menlo", 13, QFont.Weight.Bold))
        painter.drawText(24, 34, "GRASSMANN SUBSPACE GALAXY")
        painter.setPen(QColor("#8aa4b8"))
        painter.setFont(QFont("Menlo", 10))
        painter.drawText(24, 56, "Drag to rotate | trackpad/wheel or +/- to zoom | double click to reset")
        if top_match is not None:
            angles = ", ".join(f"{angle:.1f}" for angle in top_match.canonical_angles_degrees[:5])
            painter.setPen(QColor("#ffd166"))
            painter.drawText(
                24,
                self.height() - 42,
                f"Closest orbit: {top_match.label} | mean canonical angle {top_match.mean_angle_degrees:.1f} deg",
            )
            painter.setPen(QColor("#8aa4b8"))
            painter.drawText(24, self.height() - 22, f"Canonical angles: {angles or 'n/a'}")

    def _project_nodes(self) -> list[tuple[GalaxyNode, QPointF, float]]:
        rotation = self._rotation_terms()
        focal_length = min(self.width(), self.height()) * 0.84
        center = QPointF(self.width() * 0.5, self.height() * 0.52)
        rotated = []
        for node in self.nodes:
            x, y, z = self._rotate_point(node.position.astype(float), rotation)
            depth = max(0.25, self.camera_distance - z)
            perspective = focal_length / depth
            point = QPointF(center.x() + x * perspective, center.y() - y * perspective)
            rotated.append((node, point, z))
        return rotated

    def _project_starfield(self) -> list[tuple[float, float, float, float, int]]:
        rotation = self._rotation_terms()
        focal_length = min(self.width(), self.height()) * 0.84
        center_x = self.width() * 0.5
        center_y = self.height() * 0.52
        projected = []
        for point, size, alpha in self.starfield:
            x, y, z = self._rotate_point(point, rotation)
            depth = max(0.35, self.camera_distance - z)
            perspective = focal_length / depth
            screen_x = center_x + x * perspective
            screen_y = center_y - y * perspective
            if -80 <= screen_x <= self.width() + 80 and -80 <= screen_y <= self.height() + 80:
                projected.append((screen_x, screen_y, depth, size, alpha))
        return projected

    def _rotation_terms(self) -> tuple[float, float, float, float]:
        rx = math.radians(self.rotation_x)
        ry = math.radians(self.rotation_y)
        return math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry)

    def _rotate_point(
        self,
        point: np.ndarray,
        rotation: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        cos_x, sin_x, cos_y, sin_y = rotation
        x, y, z = point.astype(float)
        y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x
        x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y
        return x, y, z

    def _build_starfield(self) -> list[tuple[np.ndarray, float, int]]:
        rng = np.random.default_rng(42)
        stars = []
        for _ in range(120):
            direction = rng.normal(size=3)
            direction = direction / max(1e-9, float(np.linalg.norm(direction)))
            radius = float(rng.uniform(2.2, 4.8))
            stars.append(
                (
                    (direction * radius).astype(np.float64),
                    float(rng.uniform(0.6, 1.8)),
                    int(rng.integers(40, 145)),
                )
            )
        return stars


class SubspaceGalaxyScreen(BaseScreen):
    def __init__(self) -> None:
        super().__init__()
        self.result: dict[str, Any] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(18)

        panel = QFrame()
        panel.setObjectName("terminalPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(14)

        self.canvas = SubspaceGalaxyCanvas()
        self.status_label = self.make_status("Load a match result to visualize subspace distances.")
        zoom_out_button = QPushButton("[ - ZOOM ]")
        zoom_out_button.clicked.connect(self.canvas.zoom_out)
        zoom_in_button = QPushButton("[ + ZOOM ]")
        zoom_in_button.clicked.connect(self.canvas.zoom_in)
        reset_button = QPushButton("[ RESET GALAXY VIEW ]")
        reset_button.clicked.connect(self.canvas.reset_view)

        panel_layout.addWidget(self.make_title("3D SUBSPACE GALAXY"))
        panel_layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        controls.addWidget(zoom_out_button)
        controls.addWidget(zoom_in_button)
        controls.addWidget(reset_button)
        panel_layout.addLayout(controls)
        panel_layout.addWidget(self.status_label)

        root.addWidget(panel, 1)
        self.add_back_button(root)

    def set_result(self, result: dict[str, Any]) -> None:
        self.result = result
        try:
            nodes = build_galaxy_nodes(result)
        except Exception as exc:
            self.status_label.setText(f"Subspace galaxy unavailable: {exc}")
            self.canvas.set_nodes([], str(exc))
            return
        status = f"{max(0, len(nodes) - 1)} archive subspace(s) projected from canonical-angle distances."
        self.status_label.setText(status)
        self.canvas.set_nodes(nodes, status)


def build_galaxy_nodes(result: dict[str, Any]) -> list[GalaxyNode]:
    query_basis_path = _query_basis_path(result)
    if not query_basis_path:
        raise ValueError("query subspace basis path is missing")
    query_basis = np.load(query_basis_path)

    candidates = _candidate_basis_items(result)
    if not candidates:
        raise ValueError("no candidate subspace basis files were found")

    bases = [query_basis] + [basis for _, basis in candidates]
    positions = _embed_bases(bases)
    nodes = [
        GalaxyNode(
            label="YOU",
            position=positions[0],
            similarity=100.0,
            mean_angle_degrees=0.0,
            projection_distance=0.0,
            canonical_angles_degrees=(),
            color=QColor("#7dffe8"),
            is_query=True,
        )
    ]
    palette = ("#ffd166", "#ff7aa8", "#8cff9b", "#a78bfa", "#65d8ff", "#f4a261", "#e9ff70")
    for index, (match, basis) in enumerate(candidates, start=1):
        comparison = compare_subspaces(query_basis, basis)
        nodes.append(
            GalaxyNode(
                label=str(match.get("nickname", f"Match {index}")),
                position=positions[index],
                similarity=float(match.get("similarity", comparison.similarity)),
                mean_angle_degrees=float(match.get("mean_angle_degrees", comparison.mean_angle_degrees)),
                projection_distance=float(match.get("projection_distance", comparison.projection_distance)),
                canonical_angles_degrees=tuple(
                    float(angle)
                    for angle in match.get(
                        "canonical_angles_degrees",
                        comparison.canonical_angles_degrees.tolist(),
                    )
                ),
                color=QColor(palette[(index - 1) % len(palette)]),
            )
        )
    return nodes


def _query_basis_path(result: dict[str, Any]) -> str:
    query = result.get("query", {})
    if result.get("ritual_result"):
        return str(query.get("ritual", {}).get("full", {}).get("coordinate_subspace_basis_path", ""))
    return str(query.get("subspace_basis_path", ""))


def _candidate_basis_items(result: dict[str, Any]) -> list[tuple[dict[str, Any], np.ndarray]]:
    items = []
    query_basis_path = _query_basis_path(result)
    query_basis = np.load(query_basis_path) if query_basis_path else None
    for match in result.get("top_matches", [])[:8]:
        path = _candidate_basis_path(match, ritual=bool(result.get("ritual_result")))
        if not path or not Path(path).exists():
            continue
        basis = np.load(path)
        if query_basis is not None and basis.shape[0] != query_basis.shape[0]:
            continue
        items.append((match, basis))
    return items


def _candidate_basis_path(match: dict[str, Any], *, ritual: bool) -> str:
    metadata_path = str(match.get("metadata_path", ""))
    metadata: dict[str, Any] = {}
    if metadata_path and Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if ritual:
        return str(metadata.get("ritual", {}).get("full", {}).get("coordinate_subspace_basis_path", ""))
    return str(metadata.get("subspace_basis_path") or match.get("subspace_basis_path") or "")


def _embed_bases(bases: list[np.ndarray]) -> np.ndarray:
    count = len(bases)
    distances = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            distance = compare_subspaces(bases[i], bases[j]).projection_distance
            distances[i, j] = distances[j, i] = distance
    positions = _classical_mds(distances, dimensions=3)
    positions = positions - positions[0]
    max_norm = float(np.linalg.norm(positions, axis=1).max())
    if max_norm > 1e-9:
        positions = positions / max_norm
    if count == 2:
        positions[1] = np.array([0.9, 0.0, 0.0], dtype=np.float64)
    return positions


def _classical_mds(distances: np.ndarray, *, dimensions: int) -> np.ndarray:
    n = distances.shape[0]
    if n == 1:
        return np.zeros((1, dimensions), dtype=np.float64)
    squared = distances**2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    coords = np.zeros((n, dimensions), dtype=np.float64)
    usable = min(dimensions, n)
    for dim in range(usable):
        value = max(0.0, float(eigenvalues[dim]))
        coords[:, dim] = eigenvectors[:, dim] * math.sqrt(value)
    return coords
