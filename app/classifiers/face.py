# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
import mediapipe as mp
from ..utils.image_io import to_rgb

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def _P(lm, idx, w, h):
    p = lm[idx]
    return np.array([p.x * w, p.y * h])

def draw_debug(bgr: np.ndarray, res) -> np.ndarray:
    """FaceMesh 랜드마크/테셀레이션을 원본 위에 그려서 반환"""
    out = bgr.copy()
    rgb = to_rgb(out)
    if res and res.multi_face_landmarks:
        for face_lms in res.multi_face_landmarks:
            mp_draw.draw_landmarks(
                image=rgb,
                landmark_list=face_lms,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_draw.draw_landmarks(
                image=rgb,
                landmark_list=face_lms,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
            )
    # 다시 BGR로
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def classify(bgr, return_debug: bool=False) -> Tuple[Dict[str, Any], bytes]:
    
    """얼굴형 분류: round/square/oval/oblong/heart/unknown"""
    rgb = to_rgb(bgr)
    h, w = rgb.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True,
                               max_num_faces=1, min_detection_confidence=0.5) as mesh:
        res = mesh.process(rgb)
        
    if not res.multi_face_landmarks:
        return {"face_shape":"unknown", "metrics":None, "debug":"no_face"}, b""


    lm = res.multi_face_landmarks[0].landmark
    left_face  = _P(lm, 234, w, h)
    right_face = _P(lm, 454, w, h)
    face_width = np.linalg.norm(right_face - left_face)

    forehead = _P(lm, 10, w, h)
    chin     = _P(lm, 152, w, h)
    face_length = np.linalg.norm(chin - forehead)

    brow_left  = _P(lm, 70, w, h)
    brow_right = _P(lm, 300, w, h)
    forehead_width = np.linalg.norm(brow_right - brow_left)
    jaw_width = face_width * 0.9  # 근사

    ratio_len_width = face_length / (face_width + 1e-6)
    ratio_jaw_forehead = jaw_width / (forehead_width + 1e-6)

    fs = "oval"
    if ratio_len_width <= 1.1 and 0.95 <= ratio_jaw_forehead <= 1.05:
        fs = "round"
    elif ratio_len_width <= 1.15 and ratio_jaw_forehead > 1.05:
        fs = "square"
    elif ratio_len_width > 1.55:
        fs = "oblong"
    elif ratio_jaw_forehead < 0.9:
        fs = "heart"

    out = {
        "face_shape": fs,
        "metrics": {
            "face_width": float(face_width),
            "face_length": float(face_length),
            "forehead_width": float(forehead_width),
            "jaw_width_est": float(jaw_width),
            "ratio_len_width": float(ratio_len_width),
            "ratio_jaw_forehead": float(ratio_jaw_forehead),
        },
        "debug": "ok"
    }
    debug_png = b""
    
    if return_debug:
        dbg = draw_debug(bgr, res)
        # 바이트로는 scripts/API에서 선택 저장/반환 가능하게 처리
        success, buf = cv2.imencode(".png", dbg)
        debug_png = buf.tobytes() if success else b""
        
    return out, debug_png
