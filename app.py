import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os
import cv2
import librosa
import shutil
import uuid
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Engine Imports
from physics_engine.visualizer import generate_plotly_energy_density
from src.face_module import FaceLivenessDetector
from src.voice_module import VoiceLivenessDetector
from src.fusion import fuse_scores

# Globally initialize ML models for caching memory efficiency
@st.cache_resource
def load_ml_models():
    face_det = FaceLivenessDetector()
    voice_det = VoiceLivenessDetector()
    return face_det, voice_det

st.set_page_config(page_title="Multi-Modal Systems Integrator", layout="wide")

st.title("🌐 Multi-Modal Integrated Application")
st.markdown("Host environment for both **Biometric Exam Security** and **Theoretical Metric Mechanics**.")

# Setup dual environment tabs
tab1, tab2 = st.tabs(["🛡️ Biometric Liveness (Online Exams)", "🚀 Active Gravity Control (Metric Engine)"])

# ----------------- TAB 1: BIOMETRICS -----------------
with tab1:
    st.header("👤 Online Examination Security")
    st.markdown("Enable your webcam and microphone to analyze live deepfake/spoofing probability.")
    
    with st.spinner("Initializing Deep Learning Models (FaceMesh & Acoustic Inference)..."):
        face_detector, voice_detector = load_ml_models()
        
    class VideoProcessor:
        def __init__(self):
            self.frame_scores = []
            
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            try:
                score = face_detector.get_liveness_score(img)
                self.frame_scores.append(score)
                # Keep last 300 frames to prevent memory leaks
                if len(self.frame_scores) > 300:
                    self.frame_scores.pop(0)
                
                # Draw the dynamic score
                color = (0, 255, 0) if score > 0.5 else (0, 0, 255)
                cv2.putText(img, f"Liveness Score: {score:.2f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception:
                pass
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    class AudioProcessor:
        def __init__(self):
            self.audio_frames = []
            self.sample_rate = 48000
            
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            self.audio_frames.append(frame.to_ndarray())
            self.sample_rate = frame.sample_rate
            if len(self.audio_frames) > 500:
                self.audio_frames.pop(0)
            return frame

    webrtc_ctx = webrtc_streamer(
        key="liveness",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=VideoProcessor,
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

    if st.button("Run Liveness Fusion Validation", type="primary"):
        if not webrtc_ctx.video_processor or not webrtc_ctx.audio_processor:
            st.error("Please start the video and audio stream first via the START button above.")
        else:
            try:
                frame_scores = webrtc_ctx.video_processor.frame_scores
                audio_data = webrtc_ctx.audio_processor.audio_frames
                
                if not frame_scores:
                    st.error("No valid faces detected in the video stream yet.")
                elif not audio_data:
                    st.error("No audio captured from the microphone yet.")
                else:
                    with st.spinner("Fusing Tensor Outputs..."):
                        overall_face_score = float(np.median(frame_scores))
                        
                        # Concatenate all stored audio chunks
                        # to_ndarray() returns (channels, samples). We take the first channel.
                        y_raw = np.concatenate(audio_data, axis=1)[0, :]
                        y = y_raw.astype(np.float32) / 32768.0
                        sr = webrtc_ctx.audio_processor.sample_rate
                        
                        overall_voice_score = voice_detector.analyze_audio(y, original_sr=sr)
                        final_score, status = fuse_scores(overall_face_score, overall_voice_score)
                        
                    st.success("Verification Complete!")
                    st.metric("Final Liveness Fusion Score", f"{final_score*100:.2f}%")
                    if status == "Live":
                        st.success("Result: Authenticated (LIVE student presence confirmed)")
                    else:
                        st.error("Result: Spoofing Prevented (FAKE/Deepfake detected)")
                    st.info(f"Analyzed {len(frame_scores)} sequential video frames and {len(y)/sr:.1f}s of spectral audio data.")
                    
            except Exception as e:
                st.error(f"Error during execution: {e}")

# ----------------- TAB 2: PHYSICS -----------------
with tab2:
    st.header("🛰️ Theoretical Metric Simulator")
    st.markdown("Dynamically map the analytical stress-energy tensor structural requirements for a simulated Alcubierre warp metric.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Metric Geometry Control Module")
        v_s_val = st.slider(r"Spacecraft Velocity ($v_s$) [m/s]", min_value=1.0e7, max_value=6.0e8, value=3.0e8, step=1.0e7, format="%.1e")
        R_val = st.slider(r"Bubble Core Radius ($R$) [m]", min_value=10.0, max_value=200.0, value=50.0, step=10.0)
        sigma_val = st.slider(r"Shell Thickness ($\sigma$)", min_value=1.0, max_value=20.0, value=8.0, step=1.0)

    with col2:
        st.write(f"Generating symbolic spatial structure evaluating $v_s = {v_s_val:.2e}$, $R = {R_val}$, $\sigma = {sigma_val}$...")
        fig = generate_plotly_energy_density(v_s_val, R_val, sigma_val)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Analytical Architecture Review")
    st.markdown("The 3D geometric plot maps the exact spatial profile of the theoretical mass-energy tensor requiring **Null Energy Condition (NEC)** violation:")
    st.markdown(r"$$ T^{00} = - \frac{v_s^2}{32 \pi G} \left[ \left(\frac{\partial f}{\partial y}\right)^2 + \left(\frac{\partial f}{\partial z}\right)^2 \right] $$")
    st.markdown("This dashboard validates our analytical SymPy solver results derived in `metric_engine.py` dynamically against arbitrary spatial inputs.")
