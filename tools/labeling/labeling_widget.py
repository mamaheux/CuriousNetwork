import numpy as np

from PySide2.QtCore import Qt
from PySide2.QtGui import QBrush, QPen, QColor, QPixmap
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

LABELING_WIDTH = 1920
LABELING_HEIGHT = 1080

LABELING_STEP = 120
LABELING_TAG_WIDTH = LABELING_WIDTH // LABELING_STEP
LABELING_TAG_HEIGHT = LABELING_HEIGHT // LABELING_STEP

class LabelingWidget(QGraphicsView):
    def __init__(self):
        super().__init__()

        self._tag_brushes = []
        self._tag_brushes.append(QBrush(QColor(0, 0, 0, 0)))
        self._tag_brushes.append(QBrush(QColor(255, 0, 0, 90)))

        self._image_item = QGraphicsPixmapItem()
        self._create_default_scene()

        self.setScene(self._scene)
        self._image_item.setPixmap(
            QPixmap("/home/marc-antoine/Bureau/IFT725/CuriousNetwork/database/jpg/corridor_fresco/0-1.jpg"))

    def _create_default_scene(self):
        self._scene = QGraphicsScene()

        self._scene.addItem(self._image_item)

        line_pen = QPen(Qt.black, 2)
        for i in range(LABELING_TAG_WIDTH + 1):
            self._scene.addLine(i * LABELING_STEP, 0, i * LABELING_STEP, LABELING_HEIGHT, line_pen)

        for i in range(LABELING_TAG_HEIGHT + 1):
            self._scene.addLine(0, i * LABELING_STEP, LABELING_WIDTH, i * LABELING_STEP, line_pen)

        self._tags = np.zeros((LABELING_TAG_HEIGHT, LABELING_TAG_WIDTH), dtype=int)
        self._tags_items = []

        tag_pen = QPen(Qt.black, 0)
        for i in range(LABELING_TAG_HEIGHT):
            tags = []
            for j in range(LABELING_TAG_WIDTH):
                item = self._scene.addRect(j * LABELING_STEP, i * LABELING_STEP, LABELING_STEP, LABELING_STEP, tag_pen,
                                           self._tag_brushes[0])
                tags.append(item)

            self._tags_items.append(tags)

    def _update_tags(self):
        for i in range(LABELING_TAG_HEIGHT):
            for j in range(LABELING_TAG_WIDTH):
                self._tags_items[i][j].setBrush(self._tag_brushes[self._tags[i, j]])

    def set_image(self, path):
        self._image_item.setPixmap(QPixmap(path))

    def clear(self):
        self._tags = 0
        self._update_tags()

    def get_tags(self):
        return self._tags.copy()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        point = self.mapToScene(event.pos())
        index_x = int(point.x() / LABELING_STEP)
        index_y = int(point.y() / LABELING_STEP)

        if event.buttons() & Qt.LeftButton != 0:
            self._tags[index_y, index_x] = 1
        elif event.buttons() & Qt.RightButton != 0:
            self._tags[index_y, index_x] = 0

        self._update_tags()

    def mouseMoveEvent(self, event):
        point = self.mapToScene(event.pos())
        index_x = int(point.x() / LABELING_STEP)
        index_y = int(point.y() / LABELING_STEP)

        if event.buttons() & Qt.LeftButton != 0:
            self._tags[index_y, index_x] = 1
        elif event.buttons() & Qt.RightButton != 0:
            self._tags[index_y, index_x] = 0

        self._update_tags()