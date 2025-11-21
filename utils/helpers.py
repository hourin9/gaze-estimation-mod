import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.datasets import Gaze360, MPIIGaze

from models import (
    resnet18,
    resnet34,
    resnet50,
    mobilenet_v2,
    mobileone_s0,
    mobileone_s1,
    mobileone_s2,
    mobileone_s3,
    mobileone_s4
)


def get_model(arch, bins, pretrained=False, inference_mode=False):
    """Return the model based on the specified architecture."""
    if arch == 'resnet18':
        model = resnet18(pretrained=pretrained, num_classes=bins)
    elif arch == 'resnet34':
        model = resnet34(pretrained=pretrained, num_classes=bins)
    elif arch == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=bins)
    elif arch == "mobilenetv2":
        model = mobilenet_v2(pretrained=pretrained, num_classes=bins)
    elif arch == "mobileone_s0":
        model = mobileone_s0(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s1":
        model = mobileone_s1(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s2":
        model = mobileone_s2(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s3":
        model = mobileone_s3(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s4":
        model = mobileone_s4(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    else:
        raise ValueError(f"Please choose available model architecture, currently chosen: {arch}")
    return model


def angular_error(gaze_vector, label_vector):
    dot_product = np.dot(gaze_vector, label_vector)
    norm_product = np.linalg.norm(gaze_vector) * np.linalg.norm(label_vector)
    cosine_similarity = min(dot_product / norm_product, 0.9999999)

    return np.degrees(np.arccos(cosine_similarity))


def gaze_to_3d(gaze):
    yaw = gaze[0]   # Horizontal angle
    pitch = gaze[1]  # Vertical angle

    gaze_vector = np.zeros(3)
    gaze_vector[0] = -np.cos(pitch) * np.sin(yaw)
    gaze_vector[1] = -np.sin(pitch)
    gaze_vector[2] = -np.cos(pitch) * np.cos(yaw)

    return gaze_vector


def get_dataloader(params,  mode="train"):
    """Load dataset and return DataLoader."""

    transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if params.dataset == "gaze360":
        dataset = Gaze360(params.data, transform, angle=params.angle, binwidth=params.binwidth, mode=mode)
    elif params.dataset == "mpiigaze":
        dataset = MPIIGaze(params.data, transform, angle=params.angle, binwidth=params.binwidth)
    else:
        raise ValueError("Supported dataset are `gaze360` and `mpiigaze`")

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=params.num_workers,
        pin_memory=True
    )
    return data_loader

def draw_gaze(frame, bbox, pitch, yaw, thickness=2, color=(0, 0, 255)):
    """Draws gaze direction on a frame given bounding box and gaze angles."""
    # Unpack bounding box coordinates
    x_min, y_min, x_max, y_max = map(int, bbox[:4])

    # Calculate center of the bounding box
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    # Handle grayscale frames by converting them to BGR
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Calculate the direction of the gaze
    length = x_max - x_min
    dx = int(-length * np.sin(pitch) * np.cos(yaw))
    dy = int(-length * np.sin(yaw))

    point1 = (x_center, y_center)
    point2 = (x_center + dx, y_center + dy)

    # Draw gaze direction
    cv2.circle(frame, (x_center, y_center), radius=4, color=color, thickness=-1)
    cv2.arrowedLine(
        frame,
        point1,
        point2,
        color=color,
        thickness=thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.25
    )



def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, proportion=0.2):
    x_min, y_min, x_max, y_max = map(int, bbox[:4])

    width = x_max - x_min
    height = y_max - y_min

    corner_length = int(proportion * min(width, height))

    # Draw the rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)

    # Top-left corner
    cv2.line(image, (x_min, y_min), (x_min + corner_length, y_min), color, thickness)
    cv2.line(image, (x_min, y_min), (x_min, y_min + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x_max, y_min), (x_max - corner_length, y_min), color, thickness)
    cv2.line(image, (x_max, y_min), (x_max, y_min + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x_min, y_max), (x_min, y_max - corner_length), color, thickness)
    cv2.line(image, (x_min, y_max), (x_min + corner_length, y_max), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x_max, y_max), (x_max, y_max - corner_length), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max - corner_length, y_max), color, thickness)


def draw_bbox_gaze(
    frame: np.ndarray,
    bbox, pitch, yaw,
    boxc = (0, 255, 0), gazec = (0, 0, 255)):
    draw_bbox(frame, bbox, color=boxc)
    draw_gaze(frame, bbox, pitch, yaw, color=gazec)


def iou(boxA, boxB):
    # boxes are (x, y, w, h)
    xA = max(boxA[0], boxB[0]);
    yA = max(boxA[1], boxB[1]);
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3]);

    inter = max(0, xB - xA) * max(0, yB - yA);
    areaA = boxA[2] * boxA[3];
    areaB = boxB[2] * boxB[3];
    return inter / float(areaA + areaB - inter + 1e-6);


def xyxy2xywh(box):
    x1, y1, x2, y2 = map(int, box[:4]);
    return (x1, y1, x2 - x1, y2 - y2);


def xywh2xyxy(box):
    x, y, w, h = map(int, box[:4]);
    return (x, y, x + w, y + h);


def confidence_color(confidence: float) -> tuple[int, int, int]:
    if confidence <= 0.3:
        return (0, 255, 0);

    if confidence <= 0.75:
        return (0, 255, 255);

    return (0, 0, 255);


# Numbers in this section should be finetuned.
def cheat_rate(pitch: float, yaw: float) -> float:
    # print(f"{pitch} {yaw}");

    # Early return if student isn't looking down as no one
    # cheats via the ceiling.
    if yaw > 0.2:
        # print("Looking up");
        return 0;

    if pitch > 0.6 or pitch < -0.6:
        # print("Looking completely sideway");
        return 1;

    if pitch > 0.40 or pitch < -0.45:
        # print("Looking sideway");
        return 0.5;

    if pitch > 0.3 or pitch < -0.3:
        # print("hmm");
        return 0.25;

    return 0;

def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n

