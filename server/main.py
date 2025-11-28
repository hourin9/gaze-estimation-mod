from flask import Flask, Response;
import cv2;
import torch;
from torchvision import transforms;
import uniface;
import numpy as np;

from config import data_config
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
        except Exception as _:
            return 1;

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
face_detector = uniface.RetinaFace();

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

@app.route("/")
def test():
    return Response(
            get_webcam_frame(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        );

app.run(host="0.0.0.0", threaded=True);
cam.release();

