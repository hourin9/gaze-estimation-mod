import cv2
import torch;

from utils import helpers;

class Person:
    def __init__(self, id, bbox) -> None:
        self.id = id;
        self.pitch = 0.0;
        self.yaw = 0.0;
        self.bbox = bbox;
        self.time = 0.0;
        self.confidence = 0.0;

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

    def update_confidence(self, delta: float, decay: float = 0.1) -> None:
        rate = helpers.cheat_rate(self.pitch, self.yaw);
        if rate == 0 and self.confidence > 0:
            self.confidence -= decay;
        else:
            self.confidence += rate * delta;
            self.confidence = helpers.clamp(self.confidence, min=0, max=1);

    def draw(self, frame) -> None:
        converted_box = helpers.xywh2xyxy(self.bbox);
        color = helpers.confidence_color(self.confidence);
        helpers.draw_bbox(frame, converted_box, color=color);

        # TODO: draw with different colors
        helpers.draw_gaze(frame, converted_box, self.pitch, self.yaw)

    def is_cheating(self) -> bool:
        return self.confidence >= 0.75;

