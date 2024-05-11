from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List
import praw
from groq import Groq
import json
import logging
from fastapi.middleware.cors import CORSMiddleware

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# Initialize FastAPI app
app = FastAPI()

# Allow all origins (replace '*' with your frontend URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Reddit API setup
reddit = praw.Reddit(
    client_id="sxdoXzAYPOmUwvMcIb_zvQ",
    client_secret="X-Zut5_onmXDy_iKqONfJWYRQDyuGQ",
    user_agent="Turmerik bot by Abin"
)

# Initialize Groq client
client = Groq(
    api_key="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",
)

class RedditPost(BaseModel):
    title: str
    text: str
    comments: List[dict]
    author: str


class ScrapeResponse(BaseModel):
    response_data: List[dict]

# Function to scrape posts and comments from subreddits
def scrape_reddit_data(subreddits, health_condition):
    posts = []

    try:
        for subreddit_name in subreddits:
            subreddit = reddit.subreddit(subreddit_name)
            # Search through all submissions for the health condition
            for submission in subreddit.search(query=health_condition, sort='new', syntax='cloudsearch', limit=5):
                post_comments = [{
                    "author": str(comment.author),
                    "body": comment.body
                } for comment in submission.comments if isinstance(comment.author, praw.models.Redditor)]
                post = RedditPost(title=submission.title, text=submission.selftext, comments=post_comments, author=str(submission.author))
                posts.append(post)
    except Exception as e:
        logger.error(f"Error scraping Reddit data: {e}")

    return posts

# Function to analyze sentiment using Groq
def analyze_text_sentiment(text, health_condition):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze the following text and determine the general attitude of the person towards clinical trials and determine if the person would likely be open to participating in clinical trials related to {health_condition}:\n\n{text}\n\nThe answer to Sentiment and Openness to Clinical Trials should only be a single term out of the options given below in the reference format and SHOULD NOT insclude any explanations.Analysis should have the explanations. Provide your output only in the following json format: Sentiment: [POSITIVE/NEGATIVE/NEUTRAL],Openness to Clinical Trials: [YES/NO/MAYBE],Analysis:[the reason]"

                }
            ],
            model="llama3-70b-8192",
        )
        sentiment_analysis = chat_completion.choices[0].message.content
        sentiment_analysis = sentiment_analysis.replace("**", "")
        response_json = make_json(sentiment_analysis)
        return response_json
    except Exception as e:
        logger.error(f"Error analyzing text sentiment: {e}")
        return None

# Function to parse JSON response from sentiment analysis
def make_json(sentiment_analysis):
    """
    Extract sentiment, openness to clinical trials, and analysis from the sentiment analysis response.
    
    Args:
        sentiment_analysis (str): The response string from LLM containing sentiment analysis.
        
    Returns:
        dict: The extracted sentiment, openness to clinical trials, and analysis.
    """
    # Split the response string into lines
    lines = sentiment_analysis.split('\n')

    # Initialize variables to store sentiment, openness, and analysis
    sentiment = None
    openness_to_trials = None
    analysis = None

    # Loop through each line to extract sentiment, openness, and analysis
    for line in lines:
        if "Sentiment" in line:
            sentiment = line.split(":")[1].strip().strip('"').strip(',').rstrip('"')
        elif "Openness to Clinical Trials" in line:
            openness_to_trials = line.split(":")[1].strip().strip('"').strip(',').rstrip('"')
        elif "Analysis" in line:
            analysis = line.split(":", 1)[1].strip().strip('"').strip(',').rstrip('"')
    
    # Return sentiment, openness, and analysis as a dictionary
    return {
        "Sentiment": sentiment,
        "Openness to Clinical Trials": openness_to_trials,
        "Analysis": analysis
    }
# Function to generate personalized messages
def generate_message(text, author, sentiment, health_condition):
    try:
        prompt = f""" Given the following information:

        Text Content: {text}
        Author: {author}
        Sentiment: {sentiment}

        Generate a personalized message to engage the user about participating in a clinical trial related to {health_condition}. The message should:

        1. Mirror the user's post by referring to specific details or phrases from the text content to establish relatability.
        2. Be tailored to the user's sentiment towards clinical trials (positive, negative, or neutral).
        3. Provide relevant information about the clinical trial and its potential benefits.
        4. Encourage the user to consider participating or express willingness to learn more.
        5. It should follow an email's format : Dear {author} ,  [--message content--]  , Regards, Team Turmerik

        The output should only contain the message content and nothing else. Please make sure you donot include "Here is your personalized message" or any prefix or suffix of that sort

        """
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating message: {e}")
        return None

# Function to scrape and generate personalized messages
def scrape_and_generate_messages(subreddits: List[str], health_condition: str) -> List[dict]:
    response_data = []

    try:
        # Scrape Reddit data
        posts = scrape_reddit_data(subreddits, health_condition)

        for post in posts:
            # Analyze post sentiment
            post_text = post.title + "," + post.text
            post_sentiment = analyze_text_sentiment(post_text, health_condition)
            post_personalized_message = generate_message(post_text, post.author, post_sentiment["Analysis"], health_condition)

            # Analyze comments sentiment
            comments_data = []
            for comment in post.comments:
                comment_with_context = f"```This is the post:{post_text}.``` \n\n ```This is the comment:{comment['body']}"
                comment_sentiment = analyze_text_sentiment(comment_with_context, health_condition)
                comment_personalized_message = generate_message(comment_with_context, comment['author'], comment_sentiment["Analysis"], health_condition)
                comments_data.append({
                    "text": comment['body'],
                    "author": comment['author'],
                    "sentiment": comment_sentiment,
                    "personalized_message": comment_personalized_message
                })

            # Construct response object for the post
            post_data = {
                "title": post.title,
                "text": post.text,
                "author": post.author,
                "sentiment": post_sentiment,
                "personalized_message": post_personalized_message,
                "comments": comments_data
            }
            response_data.append(post_data)

    except Exception as e:
        logger.error(f"Error in scrape_and_generate_messages: {e}")

    return response_data


@app.get("/")
async def read_root():
    return {"message": "Hello World"}

# Route to handle scraping
@app.post("/scrape/", response_model=ScrapeResponse)
async def scrape(subreddit_data: dict):
    try:
        subreddits = subreddit_data["subreddits"]
        health_condition = subreddit_data["healthCondition"]
        response = scrape_and_generate_messages(subreddits, health_condition)
        return {"response_data": response}
    except Exception as e:
        logger.error(f"Error in /scrape/: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
