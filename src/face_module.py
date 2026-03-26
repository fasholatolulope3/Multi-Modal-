import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

class FaceLivenessDetector:
    def __init__(self):
        """
        Initialize the MediaPipe Face Landmarker (Tasks API) and temporal tracking.
        Optimized for CPU inference. This avoids the legacy `solutions` namespace 
        which is broken on certain Python 3.11/3.12 pip wheels.
        """
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            logger.info("Downloading MediaPipe face_landmarker.task model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", 
                model_path
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # EAR thresholds based on empirical data
        self.EAR_THRESHOLD = 0.21
        self.BLINK_CONSEC_FRAMES = 2
        
        # Temporal state tracking (3-second window)
        self.blink_counter = 0
        self.blink_total = 0
        self.window_start_time = time.time()
        
        # Eye landmarks indices from MediaPipe Face Mesh
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def _extract_roi(self, frame, face_landmarks) -> np.ndarray:
        """Extract the face ROI based on bounding box around all landmarks."""
        h, w, _ = frame.shape
        x_coords = [int(lm.x * w) for lm in face_landmarks]
        y_coords = [int(lm.y * h) for lm in face_landmarks]
        
        # Add padding
        pad_x = 20
        pad_y = 20
        
        x_min = max(0, min(x_coords) - pad_x)
        y_min = max(0, min(y_coords) - pad_y)
        x_max = min(w, max(x_coords) + pad_x)
        y_max = min(h, max(y_coords) + pad_y)
        
        return frame[y_min:y_max, x_min:x_max]

    def _calculate_laplacian_variance(self, image: np.ndarray) -> float:
        """Check for image blur (e.g., printed photo) using the variance of the Laplacian."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance

    def _detect_moire_patterns(self, image: np.ndarray) -> float:
        """
        Detect Moiré Patterns using Fast Fourier Transform (FFT) to spot 
        high-frequency spikes typically caused by digital screens.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute 2D FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        # Filter out low frequencies
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        magnitude_spectrum[crow-15:crow+15, ccol-15:ccol+15] = 0
        
        noise_score = np.var(magnitude_spectrum)
        return noise_score

    def _calculate_ear(self, face_landmarks, h, w) -> float:
        """Calculate the average Eye Aspect Ratio (EAR) for both eyes."""
        def pt(index):
            lm = face_landmarks[index]
            return np.array([lm.x * w, lm.y * h])
            
        def build_eye_ear(indices):
            p1, p2, p3, p4, p5, p6 = [pt(idx) for idx in indices]
            dist_v1 = np.linalg.norm(p2 - p6)
            dist_v2 = np.linalg.norm(p3 - p5)
            dist_h = np.linalg.norm(p1 - p4)
            ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
            return ear

        left_ear = build_eye_ear(self.LEFT_EYE_INDICES)
        right_ear = build_eye_ear(self.RIGHT_EYE_INDICES)
        return (left_ear + right_ear) / 2.0

    def analyze_face_with_telemetry(self, frame: np.ndarray) -> tuple[float, dict]:
        """
        Main pipeline. Fuses spatial and temporal features into a master float (0 to 1).
        Also returns detailed telemetry dictionary for Exam Proctoring.
        """
        telemetry = {
            "movement_status": "Focused",
            "multiple_faces": False,
            "no_face": False,
            "warning": "",
            "raw_features": {}
        }
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Execute Tasks API inference
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        # 1. Error Handling
        if not detection_result.face_landmarks:
            telemetry["no_face"] = True
            telemetry["warning"] = "No Face Detected"
            telemetry["movement_status"] = "Absent"
            return 0.0, telemetry
            
        if len(detection_result.face_landmarks) > 1:
            telemetry["multiple_faces"] = True
            telemetry["warning"] = "Multiple Faces Detected"
            return 0.0, telemetry
            
        face_landmarks = detection_result.face_landmarks[0]
        
        # Determine Head Pose (Yaw heuristic)
        nose = face_landmarks[1]
        left_cheek = face_landmarks[234]
        right_cheek = face_landmarks[454]
        
        dist_left = ((nose.x - left_cheek.x)**2 + (nose.y - left_cheek.y)**2)**0.5
        dist_right = ((nose.x - right_cheek.x)**2 + (nose.y - right_cheek.y)**2)**0.5
        
        if dist_left > 1.8 * dist_right:
            telemetry["movement_status"] = "Looking Right"
        elif dist_right > 1.8 * dist_left:
            telemetry["movement_status"] = "Looking Left"
            
        # 2. Extract ROI
        roi = self._extract_roi(frame, face_landmarks)
        if roi.size == 0:
            return 0.0, telemetry

        # 3. Spatial Analysis (Blur)
        lap_var = self._calculate_laplacian_variance(roi)
        blur_score = min(max((lap_var - 50.0) / 200.0, 0.0), 1.0) 

        # 4. Spatial Analysis (Moiré)
        moire_var = self._detect_moire_patterns(roi)
        moire_score = max(0.0, 1.0 - (moire_var / 5000.0)) 

        # 5. Temporal Analysis (Blink Detection)
        current_time = time.time()
        if (current_time - self.window_start_time) > 3.0:
            self.blink_total = 0
            self.window_start_time = current_time

        ear = self._calculate_ear(face_landmarks, h, w)
        
        if ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
            if telemetry["movement_status"] == "Focused":
                telemetry["movement_status"] = "Blinking"
        else:
            if self.blink_counter >= self.BLINK_CONSEC_FRAMES:
                self.blink_total += 1
            self.blink_counter = 0

        temporal_score = 1.0 if self.blink_total >= 1 else 0.5 

        # Package raw measurements for Machine Learning pipeline
        telemetry["raw_features"] = {
            "blur_score": float(blur_score),
            "moire_score": float(moire_score),
            "ear": float(ear),
            "blink_count": int(self.blink_total)
        }

        # 6. Legacy Math Fusion (Used if ML model fails/missing)
        final_liveness = (blur_score * 0.35) + (moire_score * 0.35) + (temporal_score * 0.30)
        final_score = max(0.0, min(1.0, final_liveness))
        
        if final_score < 0.5 and not telemetry["warning"]:
             telemetry["warning"] = "Suspicious Behavior Detected (Low Liveness Score)"
             
        return final_score, telemetry

    def get_liveness_score(self, frame: np.ndarray) -> float:
        """Fallback method for compatibility."""
        score, _ = self.analyze_face_with_telemetry(frame)
        return score
