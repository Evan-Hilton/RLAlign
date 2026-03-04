import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout
)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPaintEvent, QImage, QPainter, QColor
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from Agent import Agent

# -------------------- MODEL --------------------

version = ""

"""
The RLAlignModel class keeps track of all active Agents. 
Agents are defined in Agent.py
"""
class RLAlignModel:
    def __init__(self, agent_array: list[Agent]):
        self.agents = agent_array # agent_array is an array of Agent which represents the agents for each panel
        self.running = False

    def step(self):
        for agent in self.agents:
            agent.step()

    def reset(self):
        for agent in self.agents:
            agent.reset()


# -------------------- VIEW --------------------

class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.rect_x = 10

    def set_position(self, x):
        self.rect_x = x
        self.update()  # triggers repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        painter.setBrush(QColor(100, 200, 100))
        painter.drawRect(self.rect_x, 40, 50, 50)


# -------------------- CONTROLLER / MAIN WINDOW --------------------

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # set up the model to do things
        Agent1 = Agent("models/v4.0-1111", "envs/v4.0-1111", 1111)
        self.model = RLAlignModel([Agent1])

        self.setWindowTitle("Mirror Panel Alignment Tool")
        self.resize(800, 600)

        # Widgets
        self.button = QPushButton("Start / Pause")
        self.reset_button = QPushButton("Reset")

        self.plot = pg.PlotWidget()
        self.plot_curve = self.plot.plot()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.plot)
        self.setLayout(layout)

        # Timer
        self.timer = QTimer()
        self.timer.setInterval(16)  # ~60 FPS
        self.timer.timeout.connect(self.update_loop)

        # Signals
        self.button.clicked.connect(self.toggle_running)
        self.reset_button.clicked.connect(self.reset)

    # start / stop button
    def toggle_running(self):
        self.model.running = not self.model.running
        if self.model.running:
            self.timer.start()
        else:
            self.timer.stop()

    # reset button
    def reset(self):
        self.model.reset()
        self.plot_curve.setData([])

    # called at 60 Hz
    def update_loop(self):
        self.model.step()

        # Update graph
        self.plot_curve.setData(self.model.data)
    
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        img = self.model.agents[0].obs # this might be [0, 1)
        img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
        img = np.ascontiguousarray(img)
        h, w = img.shape

        qimg = QImage(
            img.data,
            w,
            h,
            img.strides[0],
            QImage.Format_Grayscale8
        )

        scaled = qimg.scaled(
            128,
            128,
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )

        painter.drawImage(0, 0, scaled)


# -------------------- APP ENTRY --------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())