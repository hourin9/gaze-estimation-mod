import cv2;

from person import Person;
from utils.helpers import iou;

DETECT_INTERVAL = 20;
MAX_MISSED = 10;

def create_tracker():
    return cv2.TrackerCSRT.create();

def trackers_update(trackers, frame):
    delete_queue = [];

    for fid, info in trackers.items():
        err = info.update_tracker(frame);
        if err >= MAX_MISSED:
            delete_queue.append(fid);

    for fid in delete_queue:
        del trackers[fid];

def trackers_redetect(trackers, frame, face_detector, next_id):
    boxes, _ = face_detector.detect(frame);
    used = set();

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box[:4]);
        w, h = x_max - x_min, y_max - y_min;
        new_box = (x_min, y_min, w, h);

        best_match = None;
        best_iou = 0.0;

        for fid, info in trackers.items():
            iou_val = iou(info.get_bbox(), new_box);
            if iou_val > best_iou:
                best_iou = iou_val;
                best_match = fid;

        if best_iou > 0.3:
            tracker = create_tracker();
            tracker.init(frame, new_box);
            trackers[best_match].bbox = new_box;
            trackers[best_match].attach_tracker(tracker);

            used.add(best_match);

        else:
            person = Person(next_id, new_box);
            tracker = create_tracker();
            tracker.init(frame, new_box);
            person.attach_tracker(tracker);
            trackers[next_id] = person;
            next_id += 1;

    return next_id;

