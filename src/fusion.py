"""
Decision level score fusion logic
"""
import logging

logger = logging.getLogger(__name__)

def fuse_scores(face_score: float, voice_score: float) -> tuple[float, str]:
    """
    Fuses the face and voice liveness scores using a weighted sum.
    We grant a 60% weight to visual liveness and 40% weight to audio liveness based 
    on the mathematical blueprint.
    
    Returns:
        tuple (final_score, status_message)
    """
    # 0.0 generally implies definitive spoof due to missing face/silent audio handling in child modules
    if face_score == 0.0 or voice_score == 0.0:
        logger.warning("Critical spoof threshold hit on one of the sensors. Denying access.")
        return 0.0, "FAILED - Critical Spoof Detected"

    # Weighted Score Fusion
    w_f = 0.60
    w_v = 0.40
    
    final_score = (face_score * w_f) + (voice_score * w_v)
    final_score = max(0.0, min(1.0, final_score))
    
    # 80% confidence threshold for "SUCCESS"
    if final_score >= 0.8:
        return final_score, "SUCCESS"
    else:
        return final_score, "FAILED - Low Confidence"

