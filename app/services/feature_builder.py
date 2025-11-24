# app/services/feature_builder.py
'''
    세 분류기(classifiers/face.py, body.py, skin.py)의 output을 통합해 
    “라벨 + 연속형 metrics + 신뢰도” 기반 feature vector 생성.
    (좀 더 자세한 추천을 위함)
'''
import numpy as np

def build_feature_vector(face, body, skin, stats, label_maps):
    """모든 분류기 결과를 하나의 수치 벡터로 결합"""
    def z(x, mu, sigma, eps=1e-8):
        return (x - mu) / (sigma + eps)

    f, lab = [], []

    # ---- body metrics ----
    m = body.get("metrics", {})
    f += [
        z(m.get("s_h", 0), *stats["s_h"]),
        z(m.get("w_s", 0), *stats["w_s"]),
        z(m.get("w_h", 0), *stats["w_h"]),
    ]
    lab += label_maps["body_shape"].get(body.get("body_shape", "unknown"), [0]*len(label_maps["body_shape"]))

    # ---- face metrics ----
    m = face.get("metrics", {})
    f += [
        z(m.get("ratio_len_width", 0), *stats["ratio_len_width"]),
        z(m.get("ratio_jaw_forehead", 0), *stats["ratio_jaw_forehead"]),
    ]
    lab += label_maps["face_shape"].get(face.get("face_shape", "unknown"), [0]*len(label_maps["face_shape"]))

    # ---- skin metrics ----
    m = skin.get("metrics", {})
    f += [
        z(m.get("L_mean", 0), *stats["L_mean"]),
        z(m.get("a_mean", 0), *stats["a_mean"]),
        z(m.get("b_mean", 0), *stats["b_mean"]),
        z(m.get("H_mean", 0), *stats["H_mean"]),
        z(m.get("S_mean", 0), *stats["S_mean"]),
        z(m.get("V_mean", 0), *stats["V_mean"]),
    ]
    lab += label_maps["skin_tone"].get(skin.get("skin_tone", "unknown"), [0]*len(label_maps["skin_tone"]))

    vec = np.array(f + lab, dtype=float)
    return vec / (np.linalg.norm(vec) + 1e-8)
