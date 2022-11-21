# Project-Corynorhinus
To test sentence embeddings from Sentence-Transformers - a POC

The project tries to answer the question: Given a tweet, when you already have a dataset of tweets, find _k_ tweets which are contextually similar to the given tweet.

# Dataset

I've taken the Donald Trump's Tweets DataSet from [DataWorld here](https://data.world/lovesdata/trump-tweets-5-4-09-12-5-16) which was created by [Julian Parris](https://data.world/lovesdata). It contains his tweets from 2009 to 2017.

Due to storage limitations on HEROKU (the platform on which this APP is hosted), I have taken a dataset of only 297 tweets. Surprisingly, it does show some good results.

# How to run?

+ Create an environment using the requirements.txt file in this repository
+ Run the python app by command `python main.py`
+ Head over to the swagger link to the getSimilarTweets endpoint
+ Click on 'Try it out'
+ Enter some number like "3" for the number of contexually similar tweets that we want [The number should be between 1 and 10]
+ Hit on 'execute'

# Project File Structure

```
.
├── Procfile
├── README.md
├── core
│   └── similar_tweets.py
├── data_processing
│   └── embedding_preparation.ipynb
├── main.py
├── processed_data
│   ├── id_embedding_mapping.pkl
│   └── sentence_id_mapping.json
└── requirements.txt
```

oh, and why Corynorhinus? It's a bat species, and also the name of a score track composed by Hanz Zimmer for the movie Batman Begins. It marked a new chapter Christian Bale's Batman's life.