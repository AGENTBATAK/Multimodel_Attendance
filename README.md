# Multimodel Attendance

Offline attendance and engagement analysis for a webcam or local video file.

## Requirements

- Python 3.9+
- Webcam or local video file
- Optional microphone or audio file
- Optional local Ollama instance for teaching suggestions

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade --force-reinstall -r requirements.txt
```

## Run

Terminal runner:

```powershell
python app.py --source 0
```

Dashboard:

```powershell
streamlit run dashboard.py
```

The dashboard accepts webcam/video input, optional audio files processed locally with Librosa, and an optional local MediaPipe Face Landmarker `.task` model. Without a MediaPipe model, it uses OpenCV Haar cascades.

## Included modules

- Vision: face detection, centroid tracking, and sustained-presence attendance
- Audio: RMS/dBFS extraction and quiet, speech-like, or ambient-noise activity labels
- Fusion: rolling engagement score from focus, movement, and audio activity
- Recommendations: optional loopback-only Ollama teaching suggestions
- Analytics: session-level attendance and engagement summaries
