import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout
)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPainter, QColor
import pyqtgraph as pg


# -------------------- MODEL --------------------

version = ""

# An agent contains an environment (the one it was trained in), an RL agent,
# and a reward function (the one collected for every time step of the simulation)
class Agent:
    def __init__(self):
        self.model = 

class RLAlignModel:
    def __init__(self):
        self.t = 0
        self.running = False
        self.data = []

    def step(self):
        self.t += 1
        value = np.sin(self.t * 0.1)
        self.data.append(value)
        return value

    def reset(self):
        self.t = 0
        self.data = []


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
        self.model = RLAlignModel()

        self.setWindowTitle("RL Visualizer Template")
        self.resize(800, 600)

        # Widgets
        self.button = QPushButton("Start / Pause")
        self.reset_button = QPushButton("Reset")

        self.drawing = DrawingWidget()

        self.plot = pg.PlotWidget()
        self.plot_curve = self.plot.plot()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.drawing)
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
        self.drawing.set_position(10)

    def update_loop(self):
        value = self.model.step()

        # Update rectangle position
        x = 200 + 100 * value
        self.drawing.set_position(x)

        # Update graph
        self.plot_curve.setData(self.model.data)


# -------------------- APP ENTRY --------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())