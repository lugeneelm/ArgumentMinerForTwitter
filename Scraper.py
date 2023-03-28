import tweepy
import pandas as pd

# Code inspired by https://github.com/Nonnecke/ScrapingTweets.git

print("Scraping Tweets")

# Twitter developer keys
access_token = '1571106350-H7Fh1uiuODVUHGYEYyvhcTXFm4YxTHkhxfHZ3b3'
access_token_secret = '4Mn4eEuWfhn8GWBZMXEMXyFYvTbCbRob84Wq05SwmLbbs'
consumer_key = 'LuKHDPxfFb9LnPiLqEsG17GGi'
consumer_secret = 'Ap2OggQK12ZHmJKEGNWxaLAkUuryon1m3sAnvk10lqebgRfTRE'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

tweets = []

count = 1

"""Twitter will automatically sample the last 7 days of data. Depending on how many total tweets there are with the specific hashtag, keyword, handle, or key phrase that you are looking for, you can set the date back further by adding since= as one of the parameters. You can also manually add in the number of tweets you want to get back in the items() section."""

# Specify topic to scrape tweets from
for tweet in tweepy.Cursor(api.search_tweets, q="#InsulateBritain -filter:retweets", count=450, lang='en', tweet_mode='extended').items(5000):

    try:
        # Specify what data to scrape
        data = [tweet.created_at, tweet.id, tweet.full_text, tweet.user._json['screen_name'], tweet.retweet_count, tweet.favorite_count]
        data = tuple(data)
        tweets.append(data)

    except tweepy.TweepError as e:
        print(e.reason)
        continue

    except StopIteration:
        break

df = pd.DataFrame(tweets, columns = ['created_at','tweet_id', 'tweet_text', 'screen_name', 'number_retweets', 'number_likes'])

# Create CSV file from scraped data
# Specify file name here 
"""Add the path to the folder you want to save the CSV file in as well as what you want the CSV file to be named inside the single quotations"""
df.to_csv(path_or_buf = '../Datasets/ScrapedTweets.csv', index=False)
print('CSV file created')
