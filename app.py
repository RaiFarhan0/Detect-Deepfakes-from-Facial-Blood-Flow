"""
Detect Deepfakes from Facial Blood Flow — Web Application Backend (Flask)
========================================================================
Provides a clean web interface and REST API for the rPPG Deepfake Detection pipeline.
Supports video uploads, instant demo sample testing, and interactive hemodynamic visualization.
"""

import os
import uuid
import traceback
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from rppg.face_roi import FaceROIExtractor
from rppg.rppg_extractor import RPPGExtractor
from rppg.signal_processor import SignalProcessor
from rppg.pulse_analyzer import PulseAnalyzer
from rppg.spatial_coherence import SpatialCoherenceAnalyzer
from rppg.detector import DeepfakePulseDetector
from rppg.synthetic import SyntheticPulseGenerator
from rppg.visualizer import PulseVisualizer

app = Flask(__name__)
app.config["SECRET_KEY"] = "facial-blood-flow-rppg-secret-2026"
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250 MB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DEMO_FOLDER = os.path.join(BASE_DIR, "demo_samples")
MODEL_PATH = os.path.join(BASE_DIR, "rppg_model.joblib")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEMO_FOLDER, exist_ok=True)


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Video file is too large (maximum 250 MB allowed)."}), 413

# Load detector model if available
detector = DeepfakePulseDetector()
if os.path.exists(MODEL_PATH):
    try:
        detector.load_model(MODEL_PATH)
        print(f"[Facial Blood Flow Detector] Pre-trained classifier loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[Facial Blood Flow Detector] Warning: Could not load model: {e}")


def analyze_video_file(video_path, method="pos", max_frames=600):
    """
    Analyzes a video file and formats a rich JSON response for the frontend.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    extractor = FaceROIExtractor()
    raw_traces = {"forehead": [], "left_cheek": [], "right_cheek": []}
    first_rois = None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Resize large video frames to 640px width for fast and stable facial processing
        h_f, w_f = frame.shape[:2]
        if w_f > 640:
            scale = 640.0 / w_f
            frame = cv2.resize(frame, (640, int(h_f * scale)))

        rois = extractor.process_frame(frame)
        if rois and all(k in rois for k in ["forehead", "left_cheek", "right_cheek"]):
            if first_rois is None:
                first_rois = {k: [int(v) for v in rois[k]["bbox"]] for k in rois}
            for k in ["forehead", "left_cheek", "right_cheek"]:
                raw_traces[k].append(rois[k]["rgb_mean"])
        else:
            for k in ["forehead", "left_cheek", "right_cheek"]:
                if len(raw_traces[k]) > 0:
                    raw_traces[k].append(raw_traces[k][-1])
                else:
                    raw_traces[k].append(np.array([128.0, 128.0, 128.0]))

        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

    cap.release()

    n_samples = len(raw_traces["forehead"])
    if n_samples < 30:
        return {"error": "Video too short or face could not be detected consistently (minimum 1 second required)."}

    # Signal Extraction
    clean_signals = {}
    for k in ["forehead", "left_cheek", "right_cheek"]:
        arr = np.array(raw_traces[k])
        bvp_raw = RPPGExtractor.extract(arr, method=method)
        clean = SignalProcessor.clean_pulse_signal(bvp_raw, fs=fps, lowcut=0.7, highcut=3.0)
        clean_signals[k] = clean

    # Heart Rate & Spectral Analysis
    hr_info = PulseAnalyzer.estimate_heart_rate(clean_signals["forehead"], fs=fps)

    # Spatial Coherence
    coh_stats = SpatialCoherenceAnalyzer.analyze_patch_set(clean_signals, fs=fps)

    # Classification & Window Voting
    verdict_info = detector.classify_video_windows(clean_signals, fs=fps)

    # Save diagnostic visualization
    plot_filename = f"diag_{uuid.uuid4().hex[:8]}.png"
    plot_path = os.path.join(UPLOAD_FOLDER, plot_filename)
    PulseVisualizer.plot_diagnostic(
        clean_signals, fs=fps, hr_info=hr_info, coh_stats=coh_stats,
        verdict_info=verdict_info, save_path=plot_path
    )

    # Downsample waveforms for interactive frontend Chart.js (up to 300 points)
    max_chart_points = 300
    step = max(1, n_samples // max_chart_points)
    time_axis = [round(i / fps, 2) for i in range(0, n_samples, step)]
    waveforms = {
        k: [round(float(v), 3) for v in clean_signals[k][::step]]
        for k in clean_signals
    }

    # Frequency Spectrum (filtered to 40 - 180 BPM for chart)
    freqs = hr_info.get("freqs", np.array([]))
    power = hr_info.get("power", np.array([]))
    spectrum_chart = []
    if len(freqs) > 0 and len(power) > 0:
        bpms = freqs * 60.0
        mask = (bpms >= 40.0) & (bpms <= 180.0)
        # normalize power for chart
        sub_bpms = bpms[mask]
        sub_pow = power[mask]
        max_p = np.max(sub_pow) if len(sub_pow) > 0 and np.max(sub_pow) > 0 else 1.0
        norm_pow = sub_pow / max_p
        for b, p in zip(sub_bpms[::2], norm_pow[::2]):
            spectrum_chart.append({"bpm": round(float(b), 1), "power": round(float(p), 4)})

    # Coherence Matrix
    matrix = coh_stats.get("coherence_matrix", np.eye(3))
    matrix_list = [[round(float(c), 3) for c in row] for row in matrix]

    return {
        "success": True,
        "video_info": {
            "fps": round(float(fps), 2),
            "duration_sec": round(float(n_samples / fps), 2),
            "resolution": f"{width}x{height}",
            "frames_processed": n_samples,
            "first_rois": first_rois
        },
        "verdict": verdict_info["verdict"],
        "confidence": round(float(verdict_info["confidence_percent"]), 1),
        "mean_fake_prob": round(float(verdict_info["mean_fake_probability"]), 3),
        "heart_rate": {
            "bpm": round(float(hr_info["bpm"]), 1),
            "peak_freq_hz": round(float(hr_info["peak_freq_hz"]), 2),
            "snr_db": round(float(hr_info["snr_db"]), 1),
            "spectral_entropy": round(float(hr_info["spectral_entropy"]), 3)
        },
        "spatial_coherence": {
            "mean_coherence": round(float(coh_stats["mean_coherence"]), 3),
            "min_coherence": round(float(coh_stats["min_coherence"]), 3),
            "cross_correlation": round(float(coh_stats["mean_cross_correlation"]), 3),
            "mean_lag_samples": round(float(coh_stats["mean_lag_abs"]), 2),
            "bpm_dispersion": round(float(coh_stats["bpm_dispersion"]), 2),
            "roi_bpms": {k: round(float(v), 1) for k, v in coh_stats.get("roi_bpms", {}).items()}
        },
        "charts": {
            "time_axis": time_axis,
            "waveforms": waveforms,
            "spectrum": spectrum_chart,
            "coherence_matrix": matrix_list,
            "roi_names": ["Forehead", "Left Cheek", "Right Cheek"]
        },
        "diagnostic_plot_url": f"/static/uploads/{plot_filename}"
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "status": "online",
        "model_loaded": detector.is_trained,
        "algorithms": ["POS (Wang 2017)", "CHROM (de Haan 2013)", "Green Channel (Verkruysse 2008)"]
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    method = request.form.get("method", "pos").lower()
    if method not in ["pos", "chrom", "green"]:
        method = "pos"

    filename = secure_filename(f"{uuid.uuid4().hex[:8]}_{file.filename}")
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        result = analyze_video_file(save_path, method=method)
        if "error" in result:
            return jsonify(result), 400
        result["video_url"] = f"/static/uploads/{filename}"
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze_sample/<sample_type>", methods=["POST"])
def analyze_sample(sample_type):
    method = request.args.get("method", "pos").lower()
    if sample_type == "real":
        video_path = os.path.join(DEMO_FOLDER, "real.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(BASE_DIR, "real.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(DEMO_FOLDER, "sample_real_72bpm.mp4")
            if not os.path.exists(video_path):
                SyntheticPulseGenerator.generate_synthetic_video(video_path, duration_sec=10.0, fps=30, is_fake=False, bpm=72.0)
    elif sample_type == "fake":
        video_path = os.path.join(DEMO_FOLDER, "fake.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(BASE_DIR, "fake.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(DEMO_FOLDER, "sample_deepfake.mp4")
            if not os.path.exists(video_path):
                SyntheticPulseGenerator.generate_synthetic_video(video_path, duration_sec=10.0, fps=30, is_fake=True, bpm=72.0)
    else:
        return jsonify({"error": "Invalid sample type. Choose 'real' or 'fake'."}), 400

    try:
        result = analyze_video_file(video_path, method=method)
        if "error" in result:
            return jsonify(result), 400
        result["video_url"] = f"/demo_samples/{os.path.basename(video_path)}"
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/demo_samples/<path:filename>")
def serve_demo_samples(filename):
    return send_from_directory(DEMO_FOLDER, filename)


if __name__ == "__main__":
    print("=" * 60)
    print("  DETECT DEEPFAKES FROM FACIAL BLOOD FLOW — WEB SERVER")
    print("  Open your browser at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
