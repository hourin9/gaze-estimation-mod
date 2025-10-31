import numpy;

from utils import helpers;

class Person:
    def __init__(self, id, bbox: numpy.ndarray) -> None:
        self.id = id;
        self.pitch = 0.0;
        self.yaw = 0.0;
        self.bbox = bbox;
        self.time = 0.0;

    def update(self, bbox, pitch, yaw) -> None:
        self.bbox = bbox;
        self.pitch = pitch;
        self.yaw = yaw;

    def draw(self, frame) -> None:
        helpers.draw_bbox(frame, self.bbox)

        # TODO: draw with different colors
        helpers.draw_gaze(frame, self.bbox, self.pitch, self.yaw)

