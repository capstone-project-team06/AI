# app/api/dto.py
# -*- coding: utf-8 -*-
"""
DTO definitions for AI server <-> Backend communication.

- DB 테이블과 1:1 매핑되는 DTO (User, UserAnalysis, Onboarding, Celebrity, Cloth ...)
- AI 서버용 API DTO (/user/analyze-photo, /style/profile 등)

ERD 기준 테이블 이름 대응:
- User              -> UserDTO
- UserAnalysis      -> UserAnalysisDTO
- Onboarding        -> OnboardingDTO
- Celebrity         -> CelebrityDTO
- CelebrityAnalysis -> CelebrityAnalysisDTO
- Cloth             -> ClothDTO
- ClothAnalysis     -> ClothAnalysisDTO
- Recommendation    -> RecommendationDTO
- RecommendationItem-> RecommendationItemDTO
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


# ============================================
# DB 테이블 1:1 DTO
# ============================================

class UserDTO(BaseModel):
    """User 테이블에 대응"""
    id: Optional[int] = None
    onboarding_id: Optional[int] = None
    name: str
    password: str
    gender: Optional[str] = None
    height: Optional[float] = None
    weight: Optional[float] = None


class UserAnalysisDTO(BaseModel):
    """UserAnalysis 테이블에 대응"""
    id: Optional[int] = None
    user_id: Optional[int] = None
    face_shape: str
    body_shape: str
    skin_tone: str
    vector: List[float]


class OnboardingDTO(BaseModel):
    """온보딩(스타일/색상/핏 선호) 테이블에 대응"""
    id: Optional[int] = None
    styles: List[str] = []
    colors: List[str] = []
    fits: List[str] = []


class CelebrityDTO(BaseModel):
    """Celebrity 테이블에 대응"""
    id: int
    name: str
    field: Optional[str] = None      # 직업/분야 (ERD의 Field 컬럼)
    face_shape: Optional[str] = None
    body_shape: Optional[str] = None
    skin_tone: Optional[str] = None
    vector: Optional[List[float]] = None


class CelebrityAnalysisDTO(BaseModel):
    """CelebrityAnalysis 테이블에 대응"""
    id: Optional[int] = None
    celeb_id: int
    face_shape: str
    body_shape: str
    skin_tone: str
    vector: List[float]


class ClothDTO(BaseModel):
    """Cloth 테이블에 대응 (실제 상품/의상)"""
    id: int
    name: str
    url: str         # 쇼핑 링크
    image_url: str   # 상품 이미지 링크


class ClothAnalysisDTO(BaseModel):
    """ClothAnalysis 테이블에 대응 (상품 특징 정보)"""
    id: Optional[int] = None
    cloth_id: int
    category: str
    sub_category: Optional[str] = None
    style: Optional[str] = None
    color: Optional[str] = None
    fit: Optional[str] = None
    season: Optional[str] = None


class RecommendationDTO(BaseModel):
    """Recommendation 테이블에 대응 (추천 헤더)"""
    id: Optional[int] = None
    user_id: int
    celeb_id: int
    reason: Optional[str] = None


class RecommendationItemDTO(BaseModel):
    """RecommendationItem 테이블에 대응 (추천 상세: 어떤 옷을 추천했는지)"""
    id: Optional[int] = None
    recommend_id: int
    cloth_id: int


# ============================================
# AI 서버 내부/응답용 추가 DTO
# ============================================

class ReferenceImageDTO(BaseModel):
    """
    레퍼런스 이미지 정보
    - image: 실제 분석에 사용할 원본 이미지 URL
    - thumb: 썸네일 URL
    - page : 원본 페이지(컨텍스트)
    """
    image: str
    thumb: Optional[str] = None
    page: Optional[str] = None
    source: str                      # serpapi / bing / google 등
    host: Optional[str] = None       # 도메인(hostname)


class ReferenceDTO(BaseModel):
    """
    연예인 기반 추천 결과 요약
    - Recommendation & RecommendationItem 으로 저장할 때 근거가 되는 데이터
    """
    celeb_id: int                    # Celebrity.id 와 매핑
    celeb_name: str                  # Celebrity.name
    similarity_score: float
    reference_images: List[ReferenceImageDTO]
    outfit_json: Dict[str, Any]      # GPT/비전 모델이 분석한 outfit 구조
    reason: Optional[str] = None     # Recommendation.reason 에 들어갈 텍스트


# ============================================
# API DTO (/user/analyze-photo, /style/profile)
# ============================================

class UserAnalyzeResponse(BaseModel):
    """
    POST /user/analyze-photo 응답
    - UserAnalysis 테이블에 그대로 저장 가능
    """
    analysis: UserAnalysisDTO


class StyleProfileRequest(BaseModel):
    """
    POST /style/profile 요청
    - 어떤 유저의 분석 결과 + 온보딩 정보를 기준으로 스타일 프로필 요청
    """
    user_analysis: UserAnalysisDTO
    onboarding: Optional[OnboardingDTO] = None


class StyleProfileResponse(BaseModel):
    """
    POST /style/profile 응답
    - user_analysis/onboarding 은 DB 값과 호환
    - reference 는 추천 근거(연예인, 레퍼런스 이미지, outfit 분석 등)
    """
    user_analysis: UserAnalysisDTO
    onboarding: Optional[OnboardingDTO] = None
    reference: ReferenceDTO
