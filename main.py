import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional, Union
import ast
import redis
import json
import time
from functools import wraps
from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

# [Previous CachedUserRecommender class code remains exactly the same, up until main()]
def redis_cache(expiration=3600):
    """
    Decorator for caching function results in Redis
    Args:
        expiration: Cache expiration time in seconds (default: 1 hour)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate a unique cache key based on function name and arguments
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # Try to get cached result
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result is not None:
                return json.loads(cached_result)
                
            # If no cache hit, execute function and cache result
            result = func(self, *args, **kwargs)
            
            # Cache the result
            self.redis_client.setex(
                cache_key,
                expiration,
                json.dumps(result, default=list)
            )
            
            return result
        return wrapper
    return decorator

class CachedUserRecommender:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.user_profiles = {}
        self.connection_network = {}
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        
        # Cache keys for bulk data
        self.PROFILES_CACHE_KEY = "user_profiles"
        self.NETWORK_CACHE_KEY = "connection_network"
    
    def _safe_int_conversion(self, value, default=0):
        """Safely convert a value to integer, handling NaN and None"""
        try:
            if pd.isna(value):
                return default
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_string_conversion(self, value, default=''):
        """Safely convert a value to string, handling NaN and None"""
        if pd.isna(value):
            return default
        return str(value).lower()
    
    def _safe_list_conversion(self, value, default=None):
        """Safely convert a comma-separated string to list, handling NaN and None"""
        if default is None:
            default = []
        if pd.isna(value):
            return default
        return [i.strip().lower() for i in str(value).split(',') if i.strip()]
    
    def _parse_connections(self, connections) -> List[int]:
        """Parse connections data from various formats into a list of integers"""
        if pd.isna(connections):
            return []
            
        if isinstance(connections, list):
            return [int(c) for c in connections]
            
        if isinstance(connections, str):
            try:
                parsed = ast.literal_eval(connections)
                if isinstance(parsed, list):
                    return [int(c) for c in parsed]
            except (ValueError, SyntaxError):
                return [int(c.strip()) for c in connections.split(',') if c.strip()]
                
        return []

    def update_from_dataframe(self, df: pd.DataFrame):
        """Update the recommender system with data from a DataFrame and cache it"""
        # Reset existing data
        self.connection_network = {}
        self.user_profiles = {}
        
        # Process each row in the DataFrame
        for _, row in df.iterrows():
            try:
                user_id = self._safe_int_conversion(row['user_id'])
                if user_id == 0:  # Skip invalid user_ids
                    continue
                
                # Store connections
                connections = self._parse_connections(row.get('connections'))
                self.connection_network[user_id] = set(connections)
                
                # Store user profile data
                self.user_profiles[user_id] = {
                    'city': self._safe_string_conversion(row.get('city')),
                    'age': self._safe_int_conversion(row.get('age')),
                    'interests': self._safe_list_conversion(row.get('interests')),
                    'state': self._safe_string_conversion(row.get('state')),
                    'bio': self._safe_string_conversion(row.get('bio')),
                    'country': self._safe_string_conversion(row.get('country')),
                    'gender': self._safe_string_conversion(row.get('gender'))
                }
                
            except Exception as e:
                print(f"Error processing user {row.get('user_id')}: {str(e)}")
                continue
        
        # Cache the processed data
        self._cache_bulk_data()
        return self
    
    def _cache_bulk_data(self):
        """Cache user profiles and connection network data"""
        # Convert sets to lists for JSON serialization
        serializable_network = {
            str(k): list(v) for k, v in self.connection_network.items()
        }
        
        # Cache the data with a default expiration of 24 hours
        self.redis_client.setex(
            self.PROFILES_CACHE_KEY,
            86400,  # 24 hours
            json.dumps(self.user_profiles)
        )
        self.redis_client.setex(
            self.NETWORK_CACHE_KEY,
            86400,  # 24 hours
            json.dumps(serializable_network)
        )
    
    def _load_cached_data(self):
        """Load data from cache if available"""
        cached_profiles = self.redis_client.get(self.PROFILES_CACHE_KEY)
        cached_network = self.redis_client.get(self.NETWORK_CACHE_KEY)
        
        if cached_profiles and cached_network:
            self.user_profiles = json.loads(cached_profiles)
            network_data = json.loads(cached_network)
            self.connection_network = {
                int(k): set(v) for k, v in network_data.items()
            }
            return True
        return False

    @redis_cache(expiration=3600)  # Cache for 1 hour
    def calculate_similarity_score(self, user_id: int, candidate_id: int) -> Tuple[float, Dict]:
        """Calculate similarity score between two users based on multiple factors"""
        if user_id not in self.user_profiles or candidate_id not in self.user_profiles:
            return 0.0, {}
            
        user = self.user_profiles[user_id]
        candidate = self.user_profiles[candidate_id]
        
        scores = {
            'city': 1.0 if user['city'] == candidate['city'] and user['city'] != '' else 0.0,
            'age': 1.0 - min(abs(user['age'] - candidate['age']) / 50.0, 1.0) if user['age'] and candidate['age'] else 0.0,
            'interests': len(set(user['interests']) & set(candidate['interests'])) / max(len(user['interests']), 1) if user['interests'] else 0.0,
            'state': 1.0 if user['state'] == candidate['state'] and user['state'] != '' else 0.0,
            'bio': 0.5 if any(word in candidate['bio'].split() for word in user['bio'].split()) else 0.0,
            'country': 1.0 if user['country'] == candidate['country'] and user['country'] != '' else 0.0,
            'gender': 1.0 if user['gender'] == candidate['gender'] and user['gender'] != '' else 0.0
        }
        
        weights = {
            'city': 64,
            'age': 32,
            'interests': 16,
            'state': 8,
            'bio': 4,
            'country': 2,
            'gender': 1
        }
        
        total_score = sum(scores[factor] * weights[factor] for factor in weights)
        max_possible_score = sum(weights.values())
        
        return total_score / max_possible_score, scores

    @redis_cache(expiration=3600)  # Cache for 30 minutes
    def get_friends_of_friends(self, user_id: int) -> Dict:
        """Get potential recommendations based on friends-of-friends analysis"""
        if user_id not in self.connection_network:
            return {}

        direct_connections = self.connection_network[user_id]
        friend_recommendations = Counter()
        
        for friend_id in direct_connections:
            if friend_id in self.connection_network:
                friend_connections = self.connection_network[friend_id]
                for potential_friend in friend_connections:
                    if potential_friend != user_id and potential_friend not in direct_connections:
                        friend_recommendations[potential_friend] += 1
        
        return dict(friend_recommendations)

    @redis_cache(expiration=3600)  # Cache for 30 minutes
    def get_recommendations(self, user_id: int, n_recommendations: int = 5, full_response=False) -> List[Dict]:
        """Get top N recommendations for a user based on network analysis and similarity"""
        if user_id not in self.connection_network:
            return []

        mutual_recs = Counter(self.get_friends_of_friends(user_id))
        recommendations = []
        direct_connections = self.connection_network[user_id]
        
        # Process mutual connections first
        for rec_id, frequency in mutual_recs.most_common():
            if len(recommendations) >= n_recommendations:
                break
                
            similarity_score, factor_scores = self.calculate_similarity_score(user_id, rec_id)
            
            if full_response:
                recommendations.append({
                    'user_id': rec_id,
                    'mutual_connection_count': frequency,
                    'mutual_connections': [
                        friend_id for friend_id in direct_connections
                        if rec_id in self.connection_network.get(friend_id, set())
                    ],
                    'similarity_score': similarity_score,
                    'factor_scores': factor_scores
                })
            else:
                recommendations.append(rec_id)

        # If we need more recommendations, add users based on similarity
        if len(recommendations) < n_recommendations:
            remaining_users = set(self.user_profiles.keys()) - {user_id} - direct_connections
            if full_response:
                remaining_users -= {r['user_id'] for r in recommendations}
            else:
                remaining_users -= set(recommendations)
            
            similarity_scores = []
            for candidate_id in remaining_users:
                score, factor_scores = self.calculate_similarity_score(user_id, candidate_id)
                similarity_scores.append((candidate_id, score, factor_scores))
            
            for candidate_id, score, factor_scores in sorted(similarity_scores, key=lambda x: x[1], reverse=True):
                if len(recommendations) >= n_recommendations:
                    break

                if full_response:
                    recommendations.append({
                        'user_id': candidate_id,
                        'mutual_connection_count': 0,
                        'mutual_connections': [],
                        'similarity_score': score,
                        'factor_scores': factor_scores
                    })
                else:
                    recommendations.append(candidate_id)
        
        return recommendations


app = FastAPI(title="User Recommendation API")

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="The ID of the user to get recommendations for")
    n_recommendations: int = Field(default=5, ge=1, le=50, description="Number of recommendations to return")
    full_response: bool = Field(default=False, description="Whether to return detailed recommendation information")

class DetailedRecommendation(BaseModel):
    user_id: int
    mutual_connection_count: int
    mutual_connections: List[int]
    similarity_score: float
    factor_scores: Dict[str, float]

class RecommendationResponse(BaseModel):
    recommendations: Union[List[int], List[Dict]] = Field(
        ..., 
        description="List of recommended user IDs or detailed recommendation information"
    )
    # execution_time: float = Field(..., description="Time taken to generate recommendations in seconds")

# Initialize the recommender as a global variable
recommender = CachedUserRecommender(
    redis_host=os.getenv('redis_host','localhost'),
    redis_port=int(os.getenv('redis_port',6379)),
    redis_db=0
)

@app.on_event("startup")
async def startup_event():
    """Initialize the recommender with data on startup"""
    try:
        df = pd.read_csv('user_data1.csv')
        recommender.update_from_dataframe(df)
    except Exception as e:
        print(f"Error loading initial data: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get user recommendations based on the provided parameters
    
    Args:
        request: RecommendationRequest object containing:
            - user_id: ID of the user to get recommendations for
            - n_recommendations: Number of recommendations to return (default: 5)
            - full_response: Whether to return detailed recommendation information (default: False)
    
    Returns:
        RecommendationResponse object containing:
            - recommendations: List of recommended users (either IDs or detailed information)
            - execution_time: Time taken to generate recommendations
    """
    try:
        start_time = time.time()
        
        # Check if user exists
        if request.user_id not in recommender.user_profiles:
            raise HTTPException(
                status_code=404,
                detail=f"User ID {request.user_id} not found"
            )
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            user_id=request.user_id,
            n_recommendations=request.n_recommendations,
            full_response=request.full_response
        )
        
        execution_time = time.time() - start_time
        
        # Format recommendations based on full_response flag
        if request.full_response:
            formatted_recommendations = [
                {
                    "user_id": rec["user_id"],
                    "mutual_connection_count": rec["mutual_connection_count"],
                    "mutual_connections": rec["mutual_connections"],
                    "similarity_score": rec["similarity_score"],
                    "factor_scores": rec["factor_scores"]
                }
                for rec in recommendations
            ]
        else:
            formatted_recommendations = recommendations
        
        return RecommendationResponse(
            recommendations=formatted_recommendations,
            # execution_time=execution_time
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)