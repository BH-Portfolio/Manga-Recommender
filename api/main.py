"""
FastAPI application for manga recommender
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.ensemble import HybridRecommender

app = FastAPI(
    title="Manga Recommender API",
    description="Hybrid recommender system for manga",
    version="1.0.0"
)

# Load models at startup
ensemble = None

@app.on_event("startup")
async def startup_event():
    global ensemble
    print("Loading models...")
    ensemble = HybridRecommender(model_dir='models')
    print("Models loaded")


class RecommendationResponse(BaseModel):
    manga_id: int
    title: str
    score: float
    reason: Optional[str] = None


class SimilarMangaRequest(BaseModel):
    manga_id: int
    n: int = 10


class UserRecommendationRequest(BaseModel):
    user_id: str
    n: int = 10


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": ensemble is not None}


@app.post("/recommend/similar", response_model=List[RecommendationResponse])
async def get_similar_manga(request: SimilarMangaRequest):
    """
    Get manga similar to the given manga
    
    Args:
        manga_id: ID of the manga to find similar titles for
        n: number of recommendations (default: 10)
    
    Returns:
        List of recommended manga
    """
    try:
        recommendations = ensemble.recommend_similar(request.manga_id, n=request.n)
        return [RecommendationResponse(**rec) for rec in recommendations]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/recommend/user", response_model=List[RecommendationResponse])
async def get_user_recommendations(request: UserRecommendationRequest):
    """
    Get personalized recommendations for a user
    
    Args:
        user_id: ID of the user
        n: number of recommendations (default: 10)
    
    Returns:
        List of recommended manga
    """
    try:
        recommendations = ensemble.recommend_for_user(request.user_id, n=request.n)
        return [RecommendationResponse(**rec) for rec in recommendations]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/manga/{manga_id}")
async def get_manga_info(manga_id: int):
    """Get information about a manga"""
    manga = ensemble.manga_df[ensemble.manga_df['manga_id'] == manga_id]
    if len(manga) == 0:
        raise HTTPException(status_code=404, detail="Manga not found")
    
    manga_data = manga.iloc[0]
    return {
        "manga_id": int(manga_data['manga_id']),
        "title": manga_data['title'],
        "mal_score": float(manga_data.get('mal_score', 0)),
        "members": int(manga_data.get('members', 0))
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Manga Recommender API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "similar": "POST /recommend/similar",
            "user": "POST /recommend/user",
            "manga_info": "GET /manga/{manga_id}"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
