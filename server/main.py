from flask import Flask, Response;
import cv2;
import torch;
from torchvision import transforms;
import uniface;
import numpy as np;

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32);
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

