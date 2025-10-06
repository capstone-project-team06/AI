'''import os
from typing import Dict, Any, Optional
from openai import OpenAI

def build_prompt(face_shape:str, body_shape:str, skin_tone:str,
                 top:Optional[str], bottom:Optional[str]) -> str:
    return f"""
You are a professional fashion stylist AI. 
Given the user's analyzed attributes and current clothing context, recommend specific styling tips and 5 concrete outfit item suggestions.

User attributes (classified, rule-based):
- Face shape: {face_shape}
- Body shape: {body_shape}
- Skin tone: {skin_tone}

Current clothing context (optional, user-provided):
- Top: {top or "N/A"}
- Bottom: {bottom or "N/A"}

Constraints & Goals:
- Provide styling that compensates for weaknesses and enhances strengths for the given face/body/skin tone.
- Include silhouette guidance (fit/length), neckline/collar, shoulder structure, rise, hem, and accessories.
- Color guidance must reference the skin tone (undertone & depth).
- Suggest 5 concrete items with: [item name], [why it fits], [how to pair], [color alternatives].
- Output in Korean. Keep it practical and concise, with bullet points.

Return JSON with fields:
- "summary": short overview (<= 3 sentences)
- "tips": list of short actionable tips
- "items": list of 5 dicts {{ "name": ..., "why": ..., "pair": ..., "colors": [...] }}
"""

def recommend(face_shape:str, body_shape:str, skin_tone:str,
              top:Optional[str], bottom:Optional[str],
              model:str="gpt-4.1-mini", temperature:float=0.7) -> Dict[str, Any]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = build_prompt(face_shape, body_shape, skin_tone, top, bottom)
    resp = client.responses.create(
        model=model,
        input=[{"role":"user","content":prompt}],
        temperature=temperature
    )

    text = ""
    try:
        text = resp.output[0].content[0].text
    except Exception:
        text = getattr(resp, "output_text", str(resp))

    import json
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {"raw": text}
    return {"raw": text, "json": parsed}
'''
# -*- coding: utf-8 -*-
import os, json
from typing import Dict, Any, Optional
from openai import OpenAI

_MODEL_DEFAULT = os.getenv("OPENAI_RECO_MODEL", "gpt-4o-mini")

# 규칙 기반 간단 백업(키 없음/오류 시)
def _fallback_recommend(face_shape: str, body_shape: str, skin_tone: str,
                        top: Optional[str], bottom: Optional[str]) -> Dict[str, Any]:
    palette = {
        "warm": {"base": ["크림", "베이지"], "accent": ["테라코타", "올리브"], "avoid": ["차가운 블루톤"]},
        "cool": {"base": ["그레이", "네이비"], "accent": ["버건디", "보라"], "avoid": ["노란기 베이지"]},
        "neutral": {"base": ["화이트", "블랙"], "accent": ["딥그린", "카멜"], "avoid": []},
    }
    under = "neutral"
    if "_warm" in skin_tone: under = "warm"
    elif "_cool" in skin_tone: under = "cool"

    tips = []
    if face_shape in ("round", "square"):
        tips.append("상의 네크라인은 V넥/딥 V로 세로선을 만들어 얼굴형을 보완.")
    if face_shape in ("oblong",):
        tips.append("라운드넥/보트넥으로 세로 길이를 줄이고 균형 맞추기.")
    if body_shape == "inverted_triangle":
        tips.append("어깨선 미니멀, 하의에 볼륨(플리츠/와이드)로 시선 하향.")
    if body_shape == "triangle":
        tips.append("어깨에 약한 패드/숄더 디테일로 상체 보강, 상의는 스트럭처.")
    if body_shape == "hourglass":
        tips.append("허리 라인 강조(크롭/벨티드), 과한 루즈핏은 피하기.")
    if body_shape == "rectangle":
        tips.append("허리 다트/핀턱, 상하 볼륨 대비로 곡선 실루엣 만들기.")

    return {
        "summary": "규칙 기반 추천(백업)",
        "color_palette": palette.get(under, palette["neutral"]),
        "items": [
            {"category": "top",
             "suggestions": ["V넥 니트", "보트넥 티", "스트럭처드 셔츠"],
             "why": "얼굴형/체형 보완을 위한 네크라인/실루엣 반영"},
            {"category": "bottom",
             "suggestions": ["와이드 팬츠", "A라인 스커트", "핀턱 슬랙스"],
             "why": "상하 밸런스와 허리선 형성"}
        ],
        "styling_tips": tips,
        "inputs_echo": {
            "face_shape": face_shape, "body_shape": body_shape, "skin_tone": skin_tone,
            "top": top, "bottom": bottom
        }
    }

def recommend(face_shape: str, body_shape: str, skin_tone: str,
              top: Optional[str] = None, bottom: Optional[str] = None,
              model: Optional[str] = None) -> Dict[str, Any]:
    """
    GPT 기반 의상 추천.
    반환: Dict (JSON 스키마 고정)
    {
      summary: str,
      color_palette: {base:[], accent:[], avoid:[]},
      items: [{category, suggestions[], why}],
      styling_tips: [],
      inputs_echo: {...}
    }
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_recommend(face_shape, body_shape, skin_tone, top, bottom)

    client = OpenAI(api_key=api_key)
    model = model or _MODEL_DEFAULT

    system = (
        "너는 퍼스널 스타일리스트야. 사용자의 얼굴형/체형/피부톤과 상·하의 정보를 바탕으로, "
        "균형과 보정을 최우선으로 한 스타일 추천을 한국어로 제공해. "
        "반드시 JSON으로만 응답하고, 설명은 간결하게."
    )
    user = {
        "face_shape": face_shape,
        "body_shape": body_shape,
        "skin_tone": skin_tone,
        "top": top,
        "bottom": bottom
    }
    schema_hint = (
        "JSON 스키마:\n"
        "{\n"
        '  "summary": string,\n'
        '  "color_palette": {"base": [string], "accent": [string], "avoid": [string]},\n'
        '  "items": [\n'
        '    {"category": "top"|"bottom"|"outer"|"shoes"|"accessory", "suggestions": [string], "why": string}\n'
        "  ],\n"
        '  "styling_tips": [string],\n'
        '  "inputs_echo": {"face_shape": string, "body_shape": string, "skin_tone": string, "top": string|null, "bottom": string|null}\n'
        "}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.6,
            max_tokens=700,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"입력: {json.dumps(user, ensure_ascii=False)}\n{schema_hint}\n"
                                            "요구사항:\n"
                                            "- 얼굴형/체형 보정 원리(네크라인, 비율, 실루엣) 반영\n"
                                            "- 피부톤(언더톤)에 맞는 컬러 팔레트 제안 (base/accent/avoid)\n"
                                            "- 현재 상의/하의가 주어지면 그것을 바탕으로 실전 코디 대안 제시\n"
                                            "- 제안 개수: items 3~6개, tips 3~6개\n"
                                            "- JSON만 출력(주석/텍스트 금지)"},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        # 안전장치: 필드 보강
        data.setdefault("summary", "")
        data.setdefault("color_palette", {"base": [], "accent": [], "avoid": []})
        data.setdefault("items", [])
        data.setdefault("styling_tips", [])
        data["inputs_echo"] = {
            "face_shape": face_shape, "body_shape": body_shape, "skin_tone": skin_tone,
            "top": top, "bottom": bottom
        }
        return data
    except Exception:
        # API 오류 시 규칙 기반으로 fallback
        return _fallback_recommend(face_shape, body_shape, skin_tone, top, bottom)
