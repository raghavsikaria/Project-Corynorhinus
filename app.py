#########################################################
#
#   Project: Project-Corynorhinus
#   Author: Raghav Sikaria (raghav.sikaria@nyu.edu)
#   Description: To test Sentence Transforomers in action
#
#########################################################

from typing import List, Dict
from fastapi import FastAPI, Path
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel, Field
from loguru import logger

from core.similar_tweets import get_n_nearest_tweets


app_description = """

##### - Raghav Sikaria (raghav.sikaria@nyu.edu)

This is an API Documentation for Project-Corynorhinus. 

It serves as a POC to test Sentence Transformers.
The project tries to answer the question: Given a tweet, when you already have a dataset of tweets, find _k_ tweets which are contextually similar to the given tweet.

Extending my gratitude to [Julian Parris](https://data.world/lovesdata) for publishing Donald Trump's Tweets DataSet [here](https://data.world/lovesdata/trump-tweets-5-4-09-12-5-16).
"""

tags_metadata = [
    {
        "name": "Find the Similar TWEET",
        "description": "API to fetch similar tweets to a randomly chosen tweet from the TRUMP Tweets Dataset",
    }
]

app = FastAPI(
    title="Project-Corynorhinus",
    version="0.0.1",
    description=app_description,
    openapi_tags=tags_metadata,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1, "syntaxHighlist.theme": "obsidian"}
)

class SimilarTweets(BaseModel):
    cosine_similarity_score: float
    tweet: str


class SimilarTweetsResponse(BaseModel):

    randomly_selected_tweet: str = Field(description="It is a randomly selected tweet in the dataset")
    similar_tweets: List[SimilarTweets] = Field(description="Similar Tweets to the randomly chosen tweets in decreasing order of COSINE Similarity Score")

    class Config:
        schema_extra = {
            "example": {
                "randomly_selected_tweet": "This is a randomly chosen tweet",
                "similar_tweets": [
                    {"cosine_similarity_score": 0.9, "similar_tweets": "Similar Tweet 1"},
                    {"cosine_similarity_score": 0.8, "similar_tweets": "Similar Tweet 2"},
                    {"cosine_similarity_score": 0.7, "similar_tweets": "Similar Tweet 3"}
                ]
            }
        }

@app.get("/getSimilarTweets/{number_of_similar_tweets}", tags=["Find the Similar TWEET"], response_model=SimilarTweetsResponse)
async def get_similar_tweets(
    number_of_similar_tweets: int = Path(default=5, description="Number of Similar Tweets you want; to the randomly chosen tweet", le = 10, gt = 0)
):
    """
    This API gets 'n' tweets contextually similar to a randomly chosen tweet from the Donald Trump Tweets Dataset. They are returned in descending order
    of the COSINE similarity score.
    """
    return get_n_nearest_tweets(number_of_similar_tweets)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0",port=5777)
