from flask import Flask, Response;
import cv2;

app = Flask(__name__);

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

