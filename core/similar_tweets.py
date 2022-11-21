import json
import random
from typing import List
from numpy import dot
from numpy.linalg import norm
import pickle

# CONSTANTS for FILE PATHS
RELATIVE_PATH_TO_ID_EMBEDDING_MAPPING_PKL_FILE = 'processed_data/id_embedding_mapping.pkl'
RELATIVE_PATH_TO_ID_SENTENCE_MAPPING_JSON_FILE = 'processed_data/sentence_id_mapping.json'

# READING AND STORING DATA IN MEMORY for faster processing for real time requests
with open(RELATIVE_PATH_TO_ID_EMBEDDING_MAPPING_PKL_FILE, "rb") as fp:
    id_to_embedding_map = pickle.load(fp)

with open(RELATIVE_PATH_TO_ID_SENTENCE_MAPPING_JSON_FILE) as f:
    id_to_tweet_map = json.load(f)


def get_a_random_tweet_id(minimum_tweet_id: int, maximum_tweet_id: int) -> int:
    """
    Responsible for generating a random number between the given range including
    the given range numbers. Assumes that there is a tweet mapped to each of the 
    IDs in this range.
    """
    return random.randint(minimum_tweet_id, maximum_tweet_id)

def find_cosine_similarity_score_with_other_tweets(chosen_tweet_id: int) -> List[List[int]]:
    """
    Responsible for finding out cosine similarity score for all tweets
    against the randomly chosen tweet, then ordering them by decreasing
    order of their scores
    """
    cosine_scores = []
    all_tweet_ids = list(id_to_tweet_map.keys())
    embedding_for_chosen_tweet_id = id_to_embedding_map[chosen_tweet_id]

    for tweet_id in all_tweet_ids:
        if tweet_id != str(chosen_tweet_id):
            embedding_of_candidate_tweet = id_to_embedding_map[int(tweet_id)]            
            cosine_score = dot(embedding_for_chosen_tweet_id, embedding_of_candidate_tweet)/(norm(embedding_for_chosen_tweet_id)*norm(embedding_of_candidate_tweet))
            cosine_scores.append([tweet_id, cosine_score])
    
    cosine_scores.sort(key = lambda x: x[1], reverse=True)
    return cosine_scores

def get_n_nearest_tweets(n: int) -> dict:
    """
    Selects a tweet randomly from the dataset, finds it COSINE Similarity score
    with all the other tweets, selects the top N closest scores and returns them
    to the VIEW Function
    """
    all_tweet_ids = [int(item_id) for item_id in list(id_to_tweet_map.keys())]
    minimum_tweet_id = min(all_tweet_ids)
    maximum_tweet_id = max(all_tweet_ids)
    chosen_tweet_id = get_a_random_tweet_id(minimum_tweet_id, maximum_tweet_id)
    all_cosine_scores = find_cosine_similarity_score_with_other_tweets(chosen_tweet_id)
    return {
        "randomly_selected_tweet": id_to_tweet_map[str(chosen_tweet_id)],
        "similar_tweets": [
            {"cosine_similarity_score": score_item[1], "tweet": id_to_tweet_map[score_item[0]]} 
            for score_item in all_cosine_scores[:n]
        ]
    }
