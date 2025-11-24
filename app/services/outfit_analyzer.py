# app/services/outfit_analyzer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Vision 모델 (원하면 gpt-4o-mini 등으로 바꿔도 됨)
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

client = OpenAI()


def analyze_outfit_with_gpt(image_urls: List[str]) -> Dict[str, Any]:
    """
    여러 장의 코디 이미지를 GPT Vision으로 분석해서
    공통된 스타일/아이템 정보를 JSON으로 반환.

    입력:
        image_urls: 분석할 이미지 URL 리스트

    반환 예시(자유도 있음, 지금은 대략 이런 구조를 가정):
    {
      "looks": [
        {
          "image_url": "...",
          "overall_style": "미니멀 캐주얼",
          "items": [
            {"category": "outer", "name": "베이지 싱글 블레이저", "color": "베이지", "fit": "슬림"},
            {"category": "top",   "name": "화이트 티셔츠",      "color": "화이트"},
            ...
          ]
        },
        ...
      ],
      "summary": "블레이저를 활용한 여름용 스마트 캐주얼 코디들...",
    }
    """
    if not image_urls:
        return {"looks": [], "summary": "no images"}

    # system 메시지: 역할 + 출력 포맷 힌트
    OUTFIT_PROMPT = """
        당신은 패션 전문 스타일리스트 + 패션 데이터셋 라벨러입니다.
        당신의 목표는 이미지 속 의상을 사람과 기계가 모두 이해하기 좋은 정규화된 JSON 형태로 구조화하는 것입니다.

        ============================================================
        [분석 대상]
        - 인물의 전체 코디(outer, top, bottom, dress, bag, shoes, accessory)
        - 소재/핏/색감/실루엣/기장/패턴까지 포함
        - 사람이 실제로 착용한 아이템만 추출 (배경 객체는 제외)
        ============================================================

        [분석 규칙 — 반드시 준수]
        1. 이미지에서 실제로 보이는 아이템만 추출 (추측 금지)
        2. 동일 카테고리의 여러 아이템이 있을 경우 모두 나열 (예: layered top)
        3. 모든 garment 항목은 다음 필드를 포함:

            {
            "category": "outer|top|bottom|dress|shoes|bag|accessory",
            "name": "명확한 의상 이름",
            "color": "기본 단색 (white/black/navy/beige/gray 등)",
            "material": "cotton|denim|leather|linen|knit|polyester|wool 등",
            "fit": "slim|regular|oversized|relaxed",
            "pattern": "solid|stripe|check|print|none",
            "details": ["버튼", "포켓", "벨트", "카라", "지퍼" 등 디테일 리스트]
            }

        4. color/material/fit/pattern이 보이지 않으면 null 또는 "unknown" 사용
        5. category는 반드시 다음 중 하나여야 함:
        - top, bottom, outer, dress, shoes, bag, accessory

        6. 분석 결과는 하나의 JSON 객체만 출력하며, 아래 스키마를 따라야 함:

        ============================================================
        [최종 출력 JSON 스키마]
        {
        "looks": [
            {
            "overall_style": "미니멀 캐주얼 / 포멀 오피스룩 / 스트릿 / 로맨틱 등",
            "garments": [
                {
                "category": "...",
                "name": "...",
                "color": "...",
                "material": "...",
                "fit": "...",
                "pattern": "...",
                "details": ["...", "..."]
                }
            ]
            }
        ],
        "summary": "전체 코디 특징 요약"
        }
        ============================================================

        [설명하지 말고 JSON만 출력하세요.]
    """

    # user 메시지 content 구성
    user_content: List[Dict[str, Any]] = []

    # 1) 텍스트 설명
    user_text = (
        "다음 이미지들에 대해 위에서 설명한 JSON 스키마에 맞춰 분석해줘.\n"
        "이미지들은 모두 같은 연예인(또는 비슷한 사람)의 코디 참고용이야.\n"
        "각 look마다 image_url 필드에 해당 이미지 URL을 그대로 넣어줘."
    )
    user_content.append({"type": "text", "text": user_text})

    # 2) 이미지 URL들 추가 (중요: type='image_url')
    for url in image_urls:
        if not url:
            continue
        user_content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    # GPT 호출
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0.2,
        max_tokens=1200,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": OUTFIT_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    content = resp.choices[0].message.content or "{}"

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # 혹시 모델이 JSON이 아닌 걸 내보내면, 최소한 래핑해서 반환
        data = {"raw": content}

    # 안전장치: 필드 기본값 보정
    data.setdefault("looks", [])
    if not isinstance(data["looks"], list):
        data["looks"] = []

    if "summary" not in data:
        # looks를 기반으로 간단 요약 만들어 넣기
        data["summary"] = f"{len(data['looks'])}개의 코디를 분석한 결과."
        
        # 🔥 여기서부터 URL 강제 매핑
    looks = data.get("looks")
    if isinstance(looks, list):
        for idx, look in enumerate(looks):
            if idx < len(image_urls):
                # 모델이 써준 image_url은 버리고, 우리가 입력한 URL을 덮어쓴다
                look["image_url"] = image_urls[idx]

    return data


# --------------------------------------------------
# 간단 CLI 테스트용 (선택)
# --------------------------------------------------
if __name__ == "__main__":
    # 예시: 임의의 이미지 URL들로 테스트
    test_urls = [
        # 실제 패션 이미지 URL을 넣어서 테스트하면 됨
        "https://example.com/some-outfit-image1.jpg",
        "https://example.com/some-outfit-image2.jpg",
    ]
    print("[TEST] analyze_outfit_with_gpt() 실행...")
    res = analyze_outfit_with_gpt(test_urls)
    print(json.dumps(res, ensure_ascii=False, indent=2))
