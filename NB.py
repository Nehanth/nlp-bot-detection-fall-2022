###################################################################################
#
#  tweepy version 3.7.0
#  
#  last_seen.txt NEEDS to be changed the latest mention at every restart
#  
####################################################################################




import time
import requests
import json
import tweepy

#naive bayes model
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

#training naive bayed model
df = pd.read_csv('CleanDataSet.csv', encoding="latin-1")
df.isnull()
df.isnull().sum().sum()
df.dropna(inplace = True)
df.describe()
df = df.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
df.fillna(0,inplace=True)
#print(df.info())
#print(df.head())
df['bot_or_human'] = df['bot_or_human'].map({'bot': 0, 'human': 1})
X = df['text']
y = df['bot_or_human']
global cv
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier\
global clf
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
#print(classification_report(y_test, y_pred))


#api keys
consumer_key = 'iRw9naKgZVPSni8fd66M4IRN6'
consumer_secret = 'WbwU8Lsbz4NdemK3BO38JNwkgCwMryWu06luESedyWRXIbdUB9'
key = '1582906439101607937-YR9LHJPmsRLFcWw9hRLddovjtPypmW'
secret = 'UubeKb9hwTNNa1wPSeNT3GN6ql4OJR6SxoKqdX2h2mXwK'
bearer_token = "AAAAAAAAAAAAAAAAAAAAAMd4iQEAAAAAFc7XLTcRPFW%2FYExRb010QcaVCTc%3DoYQoLOHseSLTFRjboIZ3ujDKMxXdEu5UvSQjzSQNcovhE2jXNn"


#api authorization
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(key, secret)
api = tweepy.API(auth)

#last seen twitter id, to avoid duplicate replies
FILE_NAME = 'last_seen.txt'


#functions to read and write to the last seen id file
def read_last_seen(FILE_NAME):
    file_read = open(FILE_NAME, 'r')
    last_seen_id = int(file_read.read().strip())
    file_read.close()
    return last_seen_id

def store_last_seen(FILE_NAME, last_seen_id):
    file_write = open(FILE_NAME, 'w')
    file_write.write(str(last_seen_id))
    file_write.close()
    return


#reply function                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
def reply():

    for tweet in reversed(tweets):
        #only replies to tweets with the #ultimatebot in it
        if '#aisaim' in tweet.full_text.lower():
            
            convo_id = analyze_tweet(tweet.id) 
          
            url = 'https://api.twitter.com/2/tweets/{}'.format(convo_id)
            tweetNLP = getConvoTweet(url)
            
            df_new = pd.DataFrame([tweetNLP], columns=['text'])
            X = df_new['text']
            A = cv.transform(X)
            prediction = clf.predict(A)
            print(tweetNLP)
            print(prediction)

            #[0] is bot [1] is human
            print(f'The tweet predicted is: {tweetNLP} ')
            
            #human
            if (str(prediction) == '[1]'):
                
                api.update_status("@" + tweet.user.screen_name + " Our model predicts that this is a: human", tweet.id)
            
            #bot
            if (str(prediction) == '[0]'):
                
                api.report_spam(screen_name = tweet.user.screen_name, perform_block = False)
                api.update_status("@" + tweet.user.screen_name + " Our model predicts that this is a: bot and has been reported", tweet.id)
          
            #store the new tweet id
            store_last_seen(FILE_NAME, tweet.id)

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r

def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

def getConvoTweet(convoUrl):
    response = connect_to_endpoint(convoUrl)
    x = json.loads(json.dumps(response))
    y = x.get('data')
    return(y.get('text'))

#use this function to get the tweet to run through the model
def analyze_tweet(tweet_id):
    url = 'https://api.twitter.com/2/tweets?ids={}&tweet.fields=conversation_id'.format(tweet_id)
    json_response = connect_to_endpoint(url)
    #extracts conversation_id from json file
    x = json.loads(json.dumps(json_response))
    y = x.get('data')
    return (y[0]['conversation_id'])

#main
while True:
    #calls on all timeline mentions (when someone @'s the bot)
    tweets = api.mentions_timeline(read_last_seen(FILE_NAME), tweet_mode='extended')
    reply()
    
    #wait timer b4 next reply
    time.sleep(7)

