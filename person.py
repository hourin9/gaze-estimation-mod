import cv2;

from utils import helpers;

class Person:
    def __init__(self, id, bbox) -> None:
        self.id = id;
        self.pitch = 0.0;
        self.yaw = 0.0;
        self.bbox = bbox;
        self.time = 0.0;

    def get_bbox(self):
        return self.bbox;

    def attach_tracker(self, tracker: cv2.Tracker) -> None:
        # print("Attached tracker");
        self.tracker = tracker;
        self.err = 0;

    def update_tracker(self, frame) -> int:
        ok, box = self.tracker.update(frame);
        if ok:
            self.bbox = box;
            self.err = 0;
        else:
            self.err += 1;
        return self.err;

    def update_gaze(self, pitch, yaw) -> None:
        self.pitch = pitch;
        self.yaw = yaw;

    def draw(self, frame) -> None:
        helpers.draw_bbox(frame, self.bbox)

        # TODO: draw with different colors
        helpers.draw_gaze(frame, self.bbox, self.pitch, self.yaw)

