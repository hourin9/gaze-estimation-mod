import atexit
import cv2;
import datetime;
import os;

class VideoClipper:
    def __init__(self, fps: float, size: cv2.typing.Size, prefix: str) -> None:
        if os.path.exists(prefix):
            os.mkdir(prefix);
        if not os.path.isdir(prefix):
            raise IOError();

        fourcc = cv2.VideoWriter.fourcc(*"mp4v");
        t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S");
        self.out = cv2.VideoWriter(f"{prefix}/{t}.mp4", fourcc, fps, size);
        atexit.register(self.close);

    def write(self, frame) -> None:
        if self.out is not None:
            self.out.write(frame);

    def close(self) -> None:
        if self.out is not None:
            self.out.release();
            self.out = None;

