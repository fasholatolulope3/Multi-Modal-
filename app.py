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
import time

# Engine Imports
from physics_engine.visualizer import generate_plotly_energy_density
from src.face_module import FaceLivenessDetector
from src.voice_module import VoiceLivenessDetector
from src.fusion import fuse_scores
from src.session_store import update_student_telemetry, get_all_students_telemetry, submit_exam_response, get_exam_submission, delete_student_record
from src.database import init_db

# Initialize database
init_db()

# Globally initialize ML models for caching memory efficiency
@st.cache_resource
def load_ml_models():
    face_det = FaceLivenessDetector()
    voice_det = VoiceLivenessDetector()
    return face_det, voice_det

st.set_page_config(page_title="Multi-Modal Systems Integrator", layout="wide")

st.title("🌐 Multi-Modal Integrated Application")
st.markdown("Host environment for both **Biometric Exam Security** and **Theoretical Metric Mechanics**.")

role = st.sidebar.selectbox("Select Role / Portal", ["Student - Exam Portal", "Admin - Monitoring Dashboard", "Active Gravity Control"])

# ----------------- PORTAL 1: STUDENT BIOMETRICS -----------------
if role == "Student - Exam Portal":
    st.header("👤 Student Examination Portal")
    
    if "student_id" not in st.session_state:
        st.session_state["student_id"] = str(uuid.uuid4())[:8]
    if "student_registered" not in st.session_state:
        st.session_state["student_registered"] = False
        
    student_id = st.session_state["student_id"]
    
    if not st.session_state["student_registered"]:
        st.info("Please register your identity before starting the exam.")
        name_input = st.text_input("Full Name", placeholder="e.g. John Doe")
        if st.button("Register & Start Exam"):
            if name_input.strip():
                st.session_state["student_name"] = name_input.strip()
                st.session_state["student_registered"] = True
                st.rerun()
            else:
                st.error("Please enter a valid name.")
        st.stop()
        
    student_name = st.session_state["student_name"]
    st.markdown(f"**Welcome, {student_name}** | Your Session ID: `{student_id}`")
    st.markdown("Please activate your camera and begin your exam. Your head movement and focus are being monitored.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("WebRTC Camera Stream")
        with st.spinner("Initializing Deep Learning Models (FaceMesh & Acoustic Inference)..."):
            face_detector, voice_detector = load_ml_models()
            
        class VideoProcessor:
            def __init__(self):
                self.frame_scores = []
                
            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                try:
                    score, telemetry = face_detector.analyze_face_with_telemetry(img)
                    self.frame_scores.append(score)
                    if len(self.frame_scores) > 300:
                        self.frame_scores.pop(0)

                    # Update local state store for Admin Dashboard
                    update_student_telemetry(student_id, student_name, telemetry)
                    
                    # Draw the dynamic score
                    color = (0, 255, 0) if score > 0.5 else (0, 0, 255)
                    cv2.putText(img, f"Score: {score:.2f}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(img, telemetry["movement_status"], (20, 70), 
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

    with col2:
        st.subheader("Exam Response Area")
        if webrtc_ctx.state.playing:
            answer = st.text_area("Question 1: Explain the geopolitical implications of Active Gravity Control.", height=300)
            if st.button("Submit Exam"):
                submit_exam_response(student_id, student_name, answer)
                st.success("Exam Submitted Successfully! You may now close this tab.")
        else:
            st.warning("⚠️ Please click 'START' on the camera feed to unlock the exam questions.")

# ----------------- PORTAL 2: ADMIN DASHBOARD -----------------
elif role == "Admin - Monitoring Dashboard":
    st.header("👨‍🏫 Admin Monitoring Dashboard")
    
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False
        
    if not st.session_state["admin_authenticated"]:
        st.warning("🔒 This portal is restricted to authorized proctors.")
        pwd = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if pwd == "admin123":
                st.session_state["admin_authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect Password")
    else:
        st.success("Authenticated as Administrator.")
        st.markdown("Real-time biometric monitoring of all active student sessions.")
        
        col1, col2 = st.columns([10, 2])
        with col2:
            st.button("🔄 Refresh Data")
            if st.button("Logout"):
                st.session_state["admin_authenticated"] = False
                st.rerun()
                
        st.divider()
        
        telemetry_data = get_all_students_telemetry()
        if not telemetry_data:
            st.info("No active student sessions detected in the local store.")
        else:
            for s_id, data in telemetry_data.items():
                s_name = data.get("student_name", "Unknown")
                with st.expander(f"{s_name} (Session {s_id}) - Last Updated: {time.strftime('%H:%M:%S', time.localtime(data['last_updated']))}", expanded=True):
                    t = data["telemetry"]
                    
                    # Check stale connection (no update in last 10 seconds)
                    if time.time() - data['last_updated'] > 10:
                        st.warning("⚠️ Stream Disconnected or Paused")
                        
                    st.write(f"**Movement Status:** {t.get('movement_status', 'Unknown')}")
                    
                    if t.get("multiple_faces"):
                        st.error("🚨 Multiple Faces Detected in Frame!")
                    if t.get("no_face"):
                        st.error("🚨 Student is Not Present!")
                    if t.get("warning"):
                        st.error(f"🚨 {t['warning']}")
                    
                    if not any([t.get("multiple_faces"), t.get("no_face"), t.get("warning")]):
                        st.success("✅ Nominal Conditions")
                        
                    exam_data = get_exam_submission(s_id)
                    if exam_data:
                        st.info(f"📝 **Exam Submitted at:** {time.strftime('%H:%M:%S', time.localtime(exam_data['submitted_at']))}")
                        st.text_area("Student Response", exam_data["response"], height=150, disabled=True, key=f"exam_{s_id}")
                    else:
                        st.warning("⏳ Exam not yet submitted.")
                    
                    st.divider()
                    if st.button("🗑️ Delete Record", key=f"del_{s_id}"):
                        delete_student_record(s_id)
                        st.rerun()

# ----------------- PORTAL 3: PHYSICS -----------------
elif role == "Active Gravity Control":
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
