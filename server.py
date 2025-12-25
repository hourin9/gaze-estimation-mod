from flask import Flask, Response, jsonify;
import cv2;
import torch;
import torch.nn.functional as F
from torchvision import transforms;
import uniface;
import numpy as np;

from config import data_config
from person import Person
from utils import tracker
from utils import helpers
from utils.clipping import VideoClipper
from utils.helpers import get_model;

def draw_stats(frame, stats):
    x = 10
    y = 30
    line_h = 30

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    cv2.rectangle(
        frame,
        (5, 5),
        (260, 5 + line_h * 2),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Total: {stats['total']}",
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Cheating: {stats['sussy']}",
        (x, y + line_h),
        font,
        scale,
        (0, 255, 255) if stats["sussy"] > 0 else (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )

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

    def estimate(self, frame) -> tuple[int, int]:
        pitch, yaw = self.gaze_detector(frame);

        pitch_predicted = F.softmax(pitch, dim=1);
        yaw_predicted = F.softmax(yaw, dim=1);

        # Mapping from binned (0 to 90) to angles (-180 to 180)
        # or (0 to 28) to angles (-42, 42)
        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, dim=1) \
                * self.binwidth \
                - self.angle;
        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, dim=1) \
                * self.binwidth \
                - self.angle;

        # Degrees to Radians
        pitch_predicted = np.radians(pitch_predicted.cpu());
        yaw_predicted = np.radians(yaw_predicted.cpu());

        return (pitch_predicted[0], yaw_predicted[0]);

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

def get_webcam_frame():
    cam = cv2.VideoCapture(0);
    while True:
        _, img = cam.read();
        _, frame = cv2.imencode(".jpg", img);
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + frame.tobytes()
               + b'\r\n');

def get_webcam_frame_with_model_shit():
    current_stats = {
        "total": 0,
        "sussy": 0,
    };

    cam = cv2.VideoCapture(0);
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

                pitch, yaw = model.estimate(image);
                info.update_gaze(pitch, yaw);
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

            current_stats["total"] = len(faces);

            sussy_count = 0;
            for face in faces.items():
                _, person = face;
                if person.is_cheating():
                    sussy_count += 1;
            current_stats["sussy"] = sussy_count;

            draw_stats(frame, current_stats);

            _, jpeg = cv2.imencode(".jpg", frame);
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                + jpeg.tobytes()
                + b'\r\n');

@app.route("/stream")
def test():
    return Response(
            get_webcam_frame_with_model_shit(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        );

@app.route("/")
def index():
    return """
    <html>
        <head>
            <title>Ho tro phat hien gian lan trong thi cu</title>
            <style>
                body { font-family: Arial; background: #111; color: white; text-align: center; }
                img { width: 640px; border: 2px solid white; }
                button { padding: 10px 20px; }
            </style>

            <script>
                function takeScreenshot() {
                    const img = document.getElementById("stream");
                    const canvas = document.createElement("canvas");

                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;

                    const ctx = canvas.getContext("2d");
                    ctx.drawImage(img, 0, 0);

                    const link = document.createElement("a");
                    link.download = Date.now() + ".png";
                    link.href = canvas.toDataURL("image/png");
                    link.click();
                }
            </script>
        </head>
        <body>
            <h1>Webcam Monitor</h1>

            <img id="stream" src="/stream" />

            <br><br>

            <button onclick="alert('gay')">Start recording</button>
            <button onclick="takeScreenshot()">Take screenshot</button>
        </body>

    </html>
    """

app.run(host="0.0.0.0", threaded=True);

