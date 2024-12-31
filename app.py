from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import pandas as pd
from dotenv import load_dotenv
import os
from cached_recommender import CachedUserRecommender
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(
    title="User Recommendation API",
    description="API for getting user recommendations based on profile similarity and network connections",
    version="1.0.0"
)

# Initialize the recommender system with environment variables
recommender = CachedUserRecommender(
    redis_host=os.getenv('redis_host', 'localhost'),
    redis_port=int(os.getenv('redis_port', 6379)),
    redis_db=0
)

# Load initial data
try:
    df = pd.read_csv('user_data1.csv')
    recommender.update_from_dataframe(df)
except Exception as e:
    print(f"Error loading initial data: {str(e)}")

# Response models
class FactorScores(BaseModel):
    city: float
    age: float
    interests: float
    state: float
    bio: float
    country: float
    gender: float

class DetailedRecommendation(BaseModel):
    user_id: int
    mutual_connection_count: int
    mutual_connections: List[int]
    similarity_score: float
    factor_scores: FactorScores

class DetailedRecommendationsResponse(BaseModel):
    recommendations: List[DetailedRecommendation]
    request_user_id: int
    total_recommendations: int

class SimpleRecommendationsResponse(BaseModel):
    recommendations: List[int]
    request_user_id: int
    total_recommendations: int

@app.get(
    "/api/v1/recommendations/",
    response_model=Union[DetailedRecommendationsResponse, SimpleRecommendationsResponse],
    responses={
        200: {"description": "Successfully retrieved recommendations"},
        404: {"description": "User not found"},
        500: {"description": "Internal server error"}
    }
)
async def get_recommendations(
    user_id: int = Query(..., description="ID of the user to get recommendations for", ge=1),
    n_recommendations: int = Query(5, description="Number of recommendations to return", ge=1, le=50),
    full_response: bool = Query(False, description="Whether to return detailed recommendation information")
):
    """
    Get personalized user recommendations.
    
    - **user_id**: ID of the user to get recommendations for
    - **n_recommendations**: Number of recommendations to return (default: 5, max: 50)
    - **full_response**: If True, returns detailed information including similarity scores and factors
    """
    try:
        # Check if user exists
        if user_id not in recommender.user_profiles:
            raise HTTPException(
                status_code=404,
                detail=f"User with ID {user_id} not found"
            )

        # Get recommendations
        recommendations = recommender.get_recommendations(
            user_id=user_id,
            n_recommendations=n_recommendations,
            full_response=full_response
        )

        if full_response:
            return DetailedRecommendationsResponse(
                recommendations=recommendations,
                request_user_id=user_id,
                total_recommendations=len(recommendations)
            )
        else:
            return SimpleRecommendationsResponse(
                recommendations=recommendations,
                request_user_id=user_id,
                total_recommendations=len(recommendations)
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint to verify service and Redis connection status"""
    try:
        # Test Redis connection
        recommender.redis_client.ping()
        return {
            "status": "healthy",
            "redis": {
                "status": "connected",
                "host": os.getenv('redis_host', 'localhost'),
                "port": int(os.getenv('redis_port', 6379))
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv('API_PORT', 8000)),
        reload=True
    )