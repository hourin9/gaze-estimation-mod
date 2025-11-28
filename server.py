from flask import Flask, Response;
import cv2;
import torch;
import torch.nn.functional as F
from torchvision import transforms;
import uniface;
import numpy as np;

from config import data_config
from utils import tracker
from utils import helpers
from utils.clipping import VideoClipper
from utils.helpers import get_model;

class ModelCont:
    def __init__(self, model: str, weight: str):
        dataset = data_config["gaze360"];
        self.bins = dataset["bins"];
        self.binwidth = dataset["binwidth"];
        self.angle = dataset["angle"];
        self.model = model;
        self.weight = weight;

    def load_model(self, device):
        self.idx_tensor = torch.arange(
            self.bins,
            device=device,
            dtype=torch.float32);

        try:
            self.gaze_detector = get_model(
                self.model,
                self.bins,
                inference_mode=True
            );
            self.state_dict = torch.load(self.weight, map_location=device);
            self.gaze_detector.load_state_dict(self.state_dict);
            self.gaze_detector.to(device);
            self.gaze_detector.eval();
        except Exception as _:
            return 1;

app = Flask(__name__);

def pre_process(image):
    if image is None or image.size == 0:
        image = np.full((448, 448, 3), 255, dtype=np.uint8);

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch

cam = cv2.VideoCapture(0);
def get_webcam_frame():
    while True:
        _, img = cam.read();
        _, frame = cv2.imencode(".jpg", img);
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + frame.tobytes()
               + b'\r\n');

def get_webcam_frame_with_model_shit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

    model = ModelCont(
        model="mobilenetv2",
        weight="weights/mobilenetv2.pt"
    );
    model.load_model(device);
    face_detector = uniface.RetinaFace();

    next_id = 0;
    faces = {};
    frame_count = 0;

    is_recording = False;
    clipper = None;

    with torch.no_grad():
        while True:
            cheating = False;
            _, frame = cam.read()

            tracker.trackers_update(faces, frame);

            if frame_count == tracker.DETECT_INTERVAL or frame_count == 0:
                next_id = tracker.trackers_redetect(
                    faces,
                    frame,
                    face_detector,
                    next_id
                );

                frame_count = 0;

            for _, info in faces.items():
                bbox = helpers.xywh2xyxy(info.get_bbox());
                x_min, y_min, x_max, y_max = map(int, bbox);

                image = frame[y_min:y_max, x_min:x_max]
                image = pre_process(image)
                image = image.to(device)

                pitch, yaw = model.gaze_detector(image)

                pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

                # Mapping from binned (0 to 90) to angles (-180 to 180) or (0 to 28) to angles (-42, 42)
                pitch_predicted = torch.sum(pitch_predicted * model.idx_tensor, dim=1) * model.binwidth - model.angle
                yaw_predicted = torch.sum(yaw_predicted * model.idx_tensor, dim=1) * model.binwidth - model.angle

                # Degrees to Radians
                pitch_predicted = np.radians(pitch_predicted.cpu())
                yaw_predicted = np.radians(yaw_predicted.cpu())

                info.update_gaze(pitch_predicted, yaw_predicted);
                info.update_confidence(0.05);

                info.draw(frame);

                if info.is_cheating():
                    cheating = True;

            # if params.output:
            #     out.write(frame)

            if cheating and not is_recording:
                print("RECORD START");
                is_recording = True;
                fps = cam.get(cv2.CAP_PROP_FPS);
                width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH));
                height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT));
                clipper = VideoClipper(fps, (width, height), "cap");

            elif not cheating and is_recording:
                print("RECORD STOP");
                if clipper is not None:
                    clipper.close();
                clipper = None;
                is_recording = False;

            if is_recording and clipper is not None:
                clipper.write(frame);

            frame_count += 1;

            _, jpeg = cv2.imencode(".jpg", frame);
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                + jpeg.tobytes()
                + b'\r\n');

@app.route("/")
def test():
    return Response(
            get_webcam_frame_with_model_shit(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        );

app.run(host="0.0.0.0", threaded=True);
cam.release();

