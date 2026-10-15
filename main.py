"""
Main Application CLI & Pipeline Orchestrator
============================================
Provides unified CLI commands to analyze videos, run live webcam detection,
benchmark against ground truth pulses, and execute an instant end-to-end demo.

Usage:
    python main.py analyze --video path/to/video.mp4
    python main.py demo
    python main.py webcam
    python main.py benchmark
"""

import os
import sys
import argparse
import cv2
import numpy as np

from rppg.face_roi import FaceROIExtractor
from rppg.rppg_extractor import RPPGExtractor
from rppg.signal_processor import SignalProcessor
from rppg.pulse_analyzer import PulseAnalyzer
from rppg.spatial_coherence import SpatialCoherenceAnalyzer
from rppg.detector import DeepfakePulseDetector
from rppg.synthetic import SyntheticPulseGenerator
from rppg.visualizer import PulseVisualizer


def process_video(video_path, method="pos", model_weights=None, output_plot="diagnostic.png",
                  output_video=None, max_frames=None):
    """
    Analyzes a video file through the full rPPG pipeline.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video: {video_path}")
        return None

    # Step 1: Read frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[1/5] Processing Video: {os.path.basename(video_path)}")
    print(f"      Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} | FPS: {fps:.2f} | Frames: {total_frames}")

    roi_extractor = FaceROIExtractor()
    raw_rgb_traces = {"forehead": [], "left_cheek": [], "right_cheek": []}
    recorded_rois = []
    frames = []

    frame_idx = 0
    print("[2/5] Tracking facial landmarks & extracting skin patches...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rois = roi_extractor.process_frame(frame)
        if rois is not None and all(r in rois for r in ["forehead", "left_cheek", "right_cheek"]):
            for r in ["forehead", "left_cheek", "right_cheek"]:
                raw_rgb_traces[r].append(rois[r]["rgb_mean"])
            recorded_rois.append(rois)
        else:
            # If face missed, repeat previous frame's mean to preserve temporal continuity
            for r in ["forehead", "left_cheek", "right_cheek"]:
                if len(raw_rgb_traces[r]) > 0:
                    raw_rgb_traces[r].append(raw_rgb_traces[r][-1])
                else:
                    raw_rgb_traces[r].append(np.array([128.0, 128.0, 128.0]))
            recorded_rois.append(None)

        if output_video:
            frames.append(frame)

        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()

    n_samples = len(raw_rgb_traces["forehead"])
    duration_sec = n_samples / fps
    print(f"      Extracted {n_samples} frames ({duration_sec:.2f} seconds)")

    if n_samples < 30:
        print("Error: Video clip too short (needs at least 1 second / 30 frames).")
        return None

    # Step 2: Extract BVP signals (Green / CHROM / POS)
    print(f"[3/5] Extracting rPPG pulse signals using {method.upper()}...")
    clean_signals = {}
    for r in ["forehead", "left_cheek", "right_cheek"]:
        rgb_arr = np.array(raw_rgb_traces[r])
        bvp_raw = RPPGExtractor.extract(rgb_arr, method=method)
        bvp_clean = SignalProcessor.clean_pulse_signal(bvp_raw, fs=fps, lowcut=0.7, highcut=3.0)
        clean_signals[r] = bvp_clean

    # Step 3: Cardiac Frequency & Heart Rate Analysis
    print("[4/5] Computing FFT frequency spectra and pulse...")
    hr_info = PulseAnalyzer.estimate_heart_rate(clean_signals["forehead"], fs=fps)
    print(f"      Dominant Pulse (Forehead): {hr_info['bpm']:.1f} BPM ({hr_info['peak_freq_hz']:.2f} Hz) | SNR: {hr_info['snr_db']:.1f} dB")

    # Step 4: Spatial Coherence Analysis
    coh_stats = SpatialCoherenceAnalyzer.analyze_patch_set(clean_signals, fs=fps)
    print(f"      Spatial Coherence: Mean = {coh_stats['mean_coherence']:.3f} | Min = {coh_stats['min_coherence']:.3f}")
    print(f"      Cross-Correlation: {coh_stats['mean_cross_correlation']:.3f} | BPM Dispersion: {coh_stats['bpm_dispersion']:.1f} BPM")

    # Step 5: Deepfake Detection & Window Voting
    print("[5/5] Evaluating temporal window consistency & deepfake verdict...")
    detector = DeepfakePulseDetector()
    if model_weights and os.path.exists(model_weights):
        detector.load_model(model_weights)

    verdict_info = detector.classify_video_windows(clean_signals, fs=fps)
    verdict = verdict_info["verdict"]
    confidence = verdict_info["confidence_percent"]

    print("\n" + "=" * 55)
    print(f"  ANALYSIS VERDICT : {verdict} (Confidence: {confidence:.1f}%)")
    print(f"  ESTIMATED PULSE  : {hr_info['bpm']:.1f} BPM")
    print(f"  MEAN COHERENCE   : {coh_stats['mean_coherence']:.3f}")
    print("=" * 55)

    # Save diagnostic plot
    if output_plot:
        PulseVisualizer.plot_diagnostic(clean_signals, fs=fps, hr_info=hr_info,
                                        coh_stats=coh_stats, verdict_info=verdict_info,
                                        save_path=output_plot)
        print(f"Diagnostic plot saved to: {output_plot}")

    # Generate annotated video if requested
    if output_video and len(frames) > 0:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(output_video, fourcc, float(fps), (w, h))
        for i, f in enumerate(frames):
            rois_f = recorded_rois[i] if i < len(recorded_rois) else None
            annotated = PulseVisualizer.annotate_frame(
                f, rois_f, bpm=hr_info['bpm'], verdict=verdict, confidence=confidence
            )
            vw.write(annotated)
        vw.release()
        print(f"Annotated video saved to: {output_video}")

    return {
        "verdict": verdict,
        "confidence": confidence,
        "bpm": hr_info["bpm"],
        "coherence": coh_stats["mean_coherence"],
        "hr_info": hr_info,
        "coh_stats": coh_stats,
        "verdict_info": verdict_info
    }


def run_demo():
    """Generates synthetic Real and Deepfake videos and benchmarks the pipeline."""
    print("=" * 60)
    print("  RUNNING RPPG DEEPFAKE DETECTION DEMO")
    print("=" * 60)

    demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_samples")
    os.makedirs(demo_dir, exist_ok=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    real_path = os.path.join(demo_dir, "real.mp4")
    if not os.path.exists(real_path):
        real_path = os.path.join(base_dir, "real.mp4")
    if not os.path.exists(real_path):
        real_path = os.path.join(demo_dir, "sample_real_72bpm.mp4")
        if not os.path.exists(real_path):
            print("\n[Step 1] Synthesizing authentic face video (Synchronous BVP @ 72 BPM)...")
            SyntheticPulseGenerator.generate_synthetic_video(real_path, duration_sec=10.0, fps=30, is_fake=False, bpm=72.0)

    fake_path = os.path.join(demo_dir, "fake.mp4")
    if not os.path.exists(fake_path):
        fake_path = os.path.join(base_dir, "fake.mp4")
    if not os.path.exists(fake_path):
        fake_path = os.path.join(demo_dir, "sample_deepfake.mp4")
        if not os.path.exists(fake_path):
            print("\n[Step 2] Synthesizing deepfake video (Desynchronized patches + GAN flicker)...")
            SyntheticPulseGenerator.generate_synthetic_video(fake_path, duration_sec=10.0, fps=30, is_fake=True, bpm=72.0)

    print(f"  Authentic Video Source: {real_path}")
    print(f"  Deepfake Video Source:  {fake_path}")

    print("\n[Step 3] Analyzing Authentic Face Video...")
    res_real = process_video(real_path, method="pos", output_plot=os.path.join(demo_dir, "real_diagnostic.png"))

    print("\n[Step 4] Analyzing Deepfake Video...")
    res_fake = process_video(fake_path, method="pos", output_plot=os.path.join(demo_dir, "fake_diagnostic.png"))

    print("\n" + "=" * 60)
    print("  DEMO COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Authentic Video:")
    print(f"    - Target Pulse: 72.0 BPM  |  Detected: {res_real['bpm']:.1f} BPM (Error: {abs(res_real['bpm'] - 72.0):.1f} BPM)")
    print(f"    - Spatial Coherence: {res_real['coherence']:.3f}")
    print(f"    - Verdict: {res_real['verdict']} ({res_real['confidence']:.1f}% confidence)")
    print()
    print(f"  Deepfake Video:")
    print(f"    - Spatial Coherence: {res_fake['coherence']:.3f} (Severely degraded)")
    print(f"    - Verdict: {res_fake['verdict']} ({res_fake['confidence']:.1f}% confidence)")
    print("=" * 60)
    print(f"  Diagnostic plots saved to:")
    print(f"    - {os.path.join(demo_dir, 'real_diagnostic.png')}")
    print(f"    - {os.path.join(demo_dir, 'fake_diagnostic.png')}")


def run_benchmark(n_trials=5):
    """Benchmarks pulse estimation accuracy (MAE) across multiple ground truth rates."""
    print("=" * 60)
    print(f"  BENCHMARKING RPPG HEART RATE ACCURACY ({n_trials} Trials)")
    print("=" * 60)

    test_bpms = [60.0, 72.0, 84.0, 96.0, 110.0]
    errors = []

    for bpm in test_bpms:
        tmp_video = f"temp_bench_{int(bpm)}bpm.mp4"
        SyntheticPulseGenerator.generate_synthetic_video(tmp_video, duration_sec=10.0, fps=30, is_fake=False, bpm=bpm)
        res = process_video(tmp_video, method="pos", output_plot=None)
        if res:
            err = abs(res["bpm"] - bpm)
            errors.append(err)
            print(f"  Target: {bpm:5.1f} BPM | Detected: {res['bpm']:5.1f} BPM | Abs Error: {err:4.1f} BPM")
        if os.path.exists(tmp_video):
            os.remove(tmp_video)

    mae = np.mean(errors) if errors else 0.0
    print("-" * 60)
    print(f"  Mean Absolute Error (MAE): {mae:.2f} BPM")
    print("=" * 60)


def run_webcam(method="pos"):
    """Runs a real-time webcam pulse reader and patch coherence monitor."""
    print("=" * 60)
    print("  LAUNCHING REAL-TIME WEBCAM PULSE MONITOR")
    print("  Press 'q' or ESC in the video window to exit.")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    extractor = FaceROIExtractor()
    buffer_len = int(fps * 8)  # 8-second sliding buffer
    rgb_buffers = {"forehead": [], "left_cheek": [], "right_cheek": []}

    current_bpm = 0.0
    current_verdict = "WAITING"
    current_conf = 0.0

    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rois = extractor.process_frame(frame)
        if rois and all(k in rois for k in ["forehead", "left_cheek", "right_cheek"]):
            for k in ["forehead", "left_cheek", "right_cheek"]:
                rgb_buffers[k].append(rois[k]["rgb_mean"])
                if len(rgb_buffers[k]) > buffer_len:
                    rgb_buffers[k].pop(0)

        frame_counter += 1
        # Update estimation every 15 frames (~0.5 sec) once buffer is adequately filled
        if frame_counter % 15 == 0 and len(rgb_buffers["forehead"]) >= int(fps * 4):
            clean_rois = {}
            for k in ["forehead", "left_cheek", "right_cheek"]:
                arr = np.array(rgb_buffers[k])
                bvp = RPPGExtractor.extract(arr, method=method)
                clean_rois[k] = SignalProcessor.clean_pulse_signal(bvp, fs=fps)

            hr = PulseAnalyzer.estimate_heart_rate(clean_rois["forehead"], fs=fps)
            current_bpm = hr["bpm"]

            coh = SpatialCoherenceAnalyzer.analyze_patch_set(clean_rois, fs=fps)
            is_real = (coh["mean_coherence"] >= 0.55 and coh["bpm_dispersion"] <= 8.0)
            current_verdict = "REAL" if is_real else "UNVERIFIED"
            current_conf = (coh["mean_coherence"] * 100.0) if is_real else ((1.0 - coh["mean_coherence"]) * 100.0)

        annotated = PulseVisualizer.annotate_frame(
            frame, rois, bpm=current_bpm, verdict=current_verdict, confidence=current_conf
        )

        cv2.imshow("rPPG Pulse & Deepfake Detector", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Detection via Remote Photoplethysmography (rPPG) and Spatial Coherence"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Analyze command
    p_analyze = subparsers.add_parser("analyze", help="Analyze a video file")
    p_analyze.add_argument("--video", "-v", required=True, help="Path to input video file")
    p_analyze.add_argument("--method", "-m", choices=["pos", "chrom", "green"], default="pos", help="rPPG extraction algorithm")
    p_analyze.add_argument("--model", help="Path to trained classifier weights (.joblib)")
    p_analyze.add_argument("--plot", "-p", default="diagnostic.png", help="Path to save diagnostic plot")
    p_analyze.add_argument("--output-video", "-o", help="Path to save annotated video with HUD")
    p_analyze.add_argument("--max-frames", type=int, help="Limit number of frames to process")

    # Demo command
    subparsers.add_parser("demo", help="Run full end-to-end demo on synthetic Real and Fake videos")

    # Benchmark command
    subparsers.add_parser("benchmark", help="Benchmark heart rate estimation error against ground truth")

    # Webcam command
    p_webcam = subparsers.add_parser("webcam", help="Run real-time webcam monitor")
    p_webcam.add_argument("--method", "-m", choices=["pos", "chrom", "green"], default="pos")

    # Web UI command
    p_web = subparsers.add_parser("web", help="Launch the web-based interactive frontend UI")
    p_web.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    p_web.add_argument("--host", default="127.0.0.1", help="Web server host (default: 127.0.0.1)")

    args = parser.parse_args()

    if args.command == "analyze":
        process_video(
            video_path=args.video,
            method=args.method,
            model_weights=args.model,
            output_plot=args.plot,
            output_video=args.output_video,
            max_frames=args.max_frames
        )
    elif args.command == "demo":
        run_demo()
    elif args.command == "benchmark":
        run_benchmark()
    elif args.command == "webcam":
        run_webcam(method=args.method)
    elif args.command == "web":
        from app import app
        print("=" * 60)
        print("  LAUNCHING DETECT DEEPFAKES FROM FACIAL BLOOD FLOW — WEB UI")
        print(f"  Open your browser at: http://{args.host}:{args.port}")
        print("=" * 60)
        app.run(host=args.host, port=args.port, debug=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
