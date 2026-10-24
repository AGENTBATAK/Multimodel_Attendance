"""Local Streamlit dashboard for attendance and engagement analysis."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Deque, Dict, Optional, Tuple, Union

import cv2
import pandas as pd

import config
from analytics_engine import (
    AudioSample,
    CascadeFaceDetector,
    EngagementScore,
    EngagementScorer,
    FaceTracker,
    MediaPipeFaceLandmarker,
    MetricsLogger,
    TrackingUpdate,
    analyze_audio_file,
    generate_session_analytics,
    simulate_audio_amplitude,
)
from app import calculate_frame_engagement, metrics_payload, should_trigger_agent
from llm_agent import LocalLLMError, LocalTeachingAgent


def persist_upload(upload, suffix: str) -> Path:
    """Store an uploaded asset only for the duration of one dashboard action."""
    with NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(upload.getbuffer())
        return Path(temporary_file.name)


def build_source(mode: str, video_upload) -> Tuple[Union[int, str], Optional[Path]]:
    """Translate dashboard input into an OpenCV source and temporary file path."""
    if mode == "Webcam":
        return 0, None
    if video_upload is None:
        raise ValueError("Choose a video file before running analysis.")
    temporary_video = persist_upload(video_upload, Path(video_upload.name).suffix or ".mp4")
    return str(temporary_video), temporary_video


def run_analysis(
    source: Union[int, str],
    frame_limit: int,
    audio: Optional[AudioSample],
    media_pipe_model: Optional[str],
    preview,
    status,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Analyze a bounded webcam or video stream fully on the local machine."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the selected video source.")

    detector = CascadeFaceDetector()
    tracker = FaceTracker()
    scorer = EngagementScorer()
    logger = MetricsLogger(interval_seconds=0.0)
    movement_history: Dict[str, Deque[float]] = defaultdict(
        lambda: deque(maxlen=config.MOVEMENT_WINDOW_FRAMES)
    )
    landmarker = MediaPipeFaceLandmarker(media_pipe_model) if media_pipe_model else None
    latest_payload: Dict[str, object] = {}

    try:
        for frame_index in range(frame_limit):
            captured, frame = capture.read()
            if not captured:
                break
            update = tracker.update(detector.detect(frame))
            active_audio = audio or AudioSample(
                rms=0.0,
                decibels_full_scale=config.AUDIO_SILENCE_DBFS,
                normalized_level=simulate_audio_amplitude(frame_index),
                captured_at=datetime.now(timezone.utc).isoformat(),
            )
            engagement = calculate_frame_engagement(
                update,
                scorer,
                movement_history,
                active_audio.normalized_level,
            )
            if landmarker is not None:
                poses = landmarker.detect_poses(frame)
                if poses:
                    movements = [
                        movement
                        for values in movement_history.values()
                        for movement in values
                    ]
                    engagement = scorer.score(
                        [pose.focus_score for pose in poses],
                        movements,
                        active_audio.normalized_level,
                    )
            logger.append_if_due(update.tracks, engagement, active_audio)
            latest_payload = metrics_payload(update, engagement, active_audio)

            if frame_index % config.DASHBOARD_FRAME_UPDATE_INTERVAL == 0:
                preview.image(frame, channels="BGR", use_container_width=True)
                status.caption(
                    f"Frame {frame_index + 1} · engagement {engagement.score:.1f}/100 · "
                    f"active profiles {sum(track.active for track in update.tracks)}"
                )
    finally:
        capture.release()
        if landmarker is not None:
            landmarker.close()

    return logger.dataframe(), tracker.attendance_dataframe(), latest_payload


def main() -> None:
    """Render the dashboard and execute analysis only after an explicit click."""
    import streamlit as st

    st.set_page_config(page_title="Attendance Monitor", page_icon="◉", layout="wide")
    st.title("Attendance monitor")
    st.caption("Local video analysis. No cloud upload. Session labels reset when analysis stops.")

    with st.sidebar:
        st.subheader("Input")
        source_mode = st.radio("Video source", ["Webcam", "Video file"])
        video_upload = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
        audio_upload = st.file_uploader("Optional audio file", type=["wav", "mp3", "flac", "ogg"])
        model_upload = st.file_uploader("Optional local MediaPipe model", type=["task"])
        frame_limit = st.slider(
            "Frames per run",
            min_value=30,
            max_value=1_200,
            value=config.DASHBOARD_DEFAULT_MAX_FRAMES,
            step=30,
        )
        ask_agent = st.checkbox("Request local Ollama suggestions", value=False)
        run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Choose a source, then run a bounded local analysis session.")
        return

    temporary_paths = []
    try:
        source, temporary_video = build_source(source_mode, video_upload)
        if temporary_video is not None:
            temporary_paths.append(temporary_video)
        audio = None
        if audio_upload is not None:
            temporary_audio = persist_upload(audio_upload, Path(audio_upload.name).suffix or ".wav")
            temporary_paths.append(temporary_audio)
            audio = analyze_audio_file(str(temporary_audio))
        model_path = None
        if model_upload is not None:
            temporary_model = persist_upload(model_upload, ".task")
            temporary_paths.append(temporary_model)
            model_path = str(temporary_model)

        preview = st.empty()
        status = st.empty()
        metrics, attendance, payload = run_analysis(
            source=source,
            frame_limit=frame_limit,
            audio=audio,
            media_pipe_model=model_path,
            preview=preview,
            status=status,
        )
        if metrics.empty:
            st.warning("No frames were read from the selected source.")
            return

        latest = metrics.iloc[-1]
        session_analytics = generate_session_analytics(attendance, metrics)
        first, second, third = st.columns(3)
        first.metric("Engagement", f"{latest['engagement_score']:.1f}/100")
        second.metric("Present profiles", int(latest["present_profiles"]))
        third.metric("Attendance rate", f"{session_analytics['attendance_rate']:.1f}%")

        st.subheader("Engagement timeline")
        st.line_chart(metrics.set_index("timestamp")[["engagement_score", "focus_score", "movement_score"]])
        st.subheader("Attendance")
        st.dataframe(attendance, use_container_width=True, hide_index=True)
        st.caption(
            f"Mean engagement {session_analytics['mean_engagement']:.1f}/100 · "
            f"minimum engagement {session_analytics['minimum_engagement']:.1f}/100 · "
            f"audio activity {latest['audio_activity']}"
        )
        st.download_button(
            "Download session metrics CSV",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="session_metrics.csv",
            mime="text/csv",
        )

        agent_score = EngagementScore(
            score=float(payload["engagement_score"]),
            focus_score=float(payload["focus_score"]),
            movement_score=float(payload["movement_score"]),
            audio_score=float(payload["audio_score"]),
            sustained_inactivity=bool(payload["sustained_inactivity"]),
        )
        if ask_agent and should_trigger_agent(agent_score, int(payload["active_profiles"])):
            try:
                suggestions = LocalTeachingAgent().request_suggestions(payload)
                st.subheader("Local teaching suggestions")
                for suggestion in suggestions:
                    st.write(f"**{suggestion.title}** — {suggestion.action} {suggestion.rationale}")
            except LocalLLMError as error:
                st.warning(f"Local Ollama is unavailable: {error}")
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        st.error(str(error))
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
