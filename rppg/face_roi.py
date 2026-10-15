"""
Face & ROI (Region of Interest) Extraction Module
=================================================
Extracts facial landmark coordinates using MediaPipe Tasks FaceLandmarker (478 points)
with an OpenCV Haar cascade fallback. Isolates key physiological skin regions:
- Forehead
- Left Cheek
- Right Cheek
- Nose Bridge
- Chin
"""

import os
import urllib.request
import cv2
import numpy as np

# Model asset URL for MediaPipe FaceLandmarker
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEFAULT_MODEL_NAME = "face_landmarker.task"

# MediaPipe landmark indices for physiological skin patches
ROI_LANDMARK_INDICES = {
    "forehead": [10, 67, 109, 103, 284, 297, 338, 9, 8],
    "left_cheek": [117, 118, 101, 205, 50, 123, 147, 187, 216],
    "right_cheek": [346, 347, 330, 425, 280, 352, 376, 411, 436],
    "nose": [1, 2, 4, 19, 94, 168, 197],
    "chin": [152, 175, 199, 200, 18, 140, 369]
}


def ensure_mediapipe_model(target_dir=None):
    """Ensures that the face_landmarker.task model is available locally."""
    if target_dir is None:
        target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(target_dir, DEFAULT_MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"[FaceROI] Downloading MediaPipe FaceLandmarker model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("[FaceROI] Model download completed.")
    return model_path


class FaceROIExtractor:
    """Extracts facial regions of interest and per-ROI RGB averages across video frames."""

    def __init__(self, model_path=None, roi_names=None):
        if roi_names is None:
            self.roi_names = ["forehead", "left_cheek", "right_cheek"]
        else:
            self.roi_names = roi_names

        self.model_path = model_path or ensure_mediapipe_model()
        self.landmarker = None
        self._init_mediapipe()

        # Haar cascade fallback
        cascade_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

        self.face_cascade = None
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self.face_cascade = cascade

    def _init_mediapipe(self):
        """Initializes the MediaPipe FaceLandmarker Tasks API."""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[FaceROI] Warning: Failed to initialize MediaPipe ({e}). Falling back to Haar Cascade.")
            self.landmarker = None

    def _get_roi_bbox_from_landmarks(self, landmarks, roi_name, frame_shape):
        """Computes pixel bounding box for a named ROI from landmark coordinates."""
        h, w = frame_shape[:2]
        indices = ROI_LANDMARK_INDICES.get(roi_name, [])
        if not indices:
            return None

        points = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                points.append([int(lm.x * w), int(lm.y * h)])

        if not points:
            return None

        points = np.array(points)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        # Apply minor padding
        pad_x = max(2, int((x_max - x_min) * 0.15))
        pad_y = max(2, int((y_max - y_min) * 0.15))

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        if x_max <= x_min or y_max <= y_min:
            return None

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def _get_roi_bbox_from_face_rect(self, face_rect, roi_name):
        """Fallback: estimates physiological ROI bounding box from overall face bounding box."""
        fx, fy, fw, fh = face_rect
        if roi_name == "forehead":
            rx = int(fx + 0.25 * fw)
            ry = int(fy + 0.10 * fh)
            rw = int(0.50 * fw)
            rh = int(0.18 * fh)
        elif roi_name == "left_cheek":
            rx = int(fx + 0.15 * fw)
            ry = int(fy + 0.50 * fh)
            rw = int(0.25 * fw)
            rh = int(0.22 * fh)
        elif roi_name == "right_cheek":
            rx = int(fx + 0.60 * fw)
            ry = int(fy + 0.50 * fh)
            rw = int(0.25 * fw)
            rh = int(0.22 * fh)
        elif roi_name == "nose":
            rx = int(fx + 0.38 * fw)
            ry = int(fy + 0.40 * fh)
            rw = int(0.24 * fw)
            rh = int(0.20 * fh)
        elif roi_name == "chin":
            rx = int(fx + 0.35 * fw)
            ry = int(fy + 0.78 * fh)
            rw = int(0.30 * fw)
            rh = int(0.15 * fh)
        else:
            return None
        return (rx, ry, rw, rh)

    def process_frame(self, frame_bgr):
        """
        Processes a single BGR video frame.
        Returns:
            dict: {roi_name: {'rgb_mean': (R, G, B), 'bbox': (x, y, w, h)}}
            or None if no face is found.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rois = {}

        # 1. Try MediaPipe FaceLandmarker
        if self.landmarker is not None:
            try:
                import mediapipe as mp
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                res = self.landmarker.detect(mp_img)
                if res.face_landmarks and len(res.face_landmarks) > 0:
                    lms = res.face_landmarks[0]
                    for name in self.roi_names:
                        bbox = self._get_roi_bbox_from_landmarks(lms, name, frame_bgr.shape)
                        if bbox is not None:
                            bx, by, bw, bh = bbox
                            patch = frame_rgb[by:by + bh, bx:bx + bw]
                            if patch.size > 0:
                                # Mean R, G, B across patch
                                rgb_mean = np.mean(patch, axis=(0, 1))
                                rois[name] = {
                                    "rgb_mean": rgb_mean,
                                    "bbox": bbox
                                }
                    if len(rois) > 0:
                        return rois
            except Exception as e:
                pass

        # 2. Fallback to Haar Cascade
        if self.face_cascade is not None and not self.face_cascade.empty():
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    # Pick largest detected face
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    face_rect = faces[0]
                    for name in self.roi_names:
                        bbox = self._get_roi_bbox_from_face_rect(face_rect, name)
                        if bbox is not None:
                            bx, by, bw, bh = bbox
                            bx = max(0, min(w - 1, bx))
                            by = max(0, min(h - 1, by))
                            bw = min(w - bx, bw)
                            bh = min(h - by, bh)
                            patch = frame_rgb[by:by + bh, bx:bx + bw]
                            if patch.size > 0:
                                rgb_mean = np.mean(patch, axis=(0, 1))
                                rois[name] = {
                                    "rgb_mean": rgb_mean,
                                    "bbox": (bx, by, bw, bh)
                                }
                    if len(rois) > 0:
                        return rois
            except Exception:
                pass

        return None
