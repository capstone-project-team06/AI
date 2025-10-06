import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

import numpy as np
from ..utils.image_io import load_image_bgr_from_bytes
from ..classifiers import face as face_mod
from ..classifiers import body as body_mod
from ..classifiers import skin as skin_mod
from ..services.gpt import recommend

app = FastAPI(title="Modular Style Pipeline", version="0.2.0")

class AnalyzeResponse(BaseModel):
    face: Dict[str, Any]
    body: Dict[str, Any]
    skin: Dict[str, Any]
    gpt: Dict[str, Any]

@app.post("/face")
async def api_face(image: UploadFile = File(...)):
    bgr = load_image_bgr_from_bytes(await image.read())
    return JSONResponse(face_mod.classify(bgr))

@app.post("/body")
async def api_body(image: UploadFile = File(...)):
    bgr = load_image_bgr_from_bytes(await image.read())
    return JSONResponse(body_mod.classify(bgr))

@app.post("/skin")
async def api_skin(image: UploadFile = File(...)):
    bgr = load_image_bgr_from_bytes(await image.read())
    return JSONResponse(skin_mod.classify(bgr))

@app.post("/recommend")
async def api_recommend(
    face_shape: str = Form(...),
    body_shape: str = Form(...),
    skin_tone: str = Form(...),
    top: Optional[str] = Form(None),
    bottom: Optional[str] = Form(None)
):
    gpt_res = recommend(face_shape, body_shape, skin_tone, top, bottom)
    return JSONResponse(gpt_res)

@app.post("/analyze", response_model=AnalyzeResponse)
async def api_analyze(
    image: UploadFile = File(...),
    top: Optional[str] = Form(None),
    bottom: Optional[str] = Form(None)
):
    bgr = load_image_bgr_from_bytes(await image.read())
    face_res = face_mod.classify(bgr)
    body_res = body_mod.classify(bgr)
    skin_res = skin_mod.classify(bgr)

    gpt_res = recommend(
        face_res.get("face_shape","unknown"),
        body_res.get("body_shape","unknown"),
        skin_res.get("skin_tone","unknown"),
        top, bottom
    )
    return JSONResponse(AnalyzeResponse(face=face_res, body=body_res, skin=skin_res, gpt=gpt_res).model_dump())
