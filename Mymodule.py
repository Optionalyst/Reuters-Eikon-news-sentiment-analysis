import eikon as ek
import pandas as pd
from bs4 import BeautifulSoup             #for HTML to text
from datetime import timedelta, date, datetime     #import datetime as dt
import datetime as dt
import nltk  #text analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # sentiment analysis
import numpy as np
def get_news_and_sentiments(key_id, ticker_and_params,ticker,start_date, end_date):
    ek.set_app_key(key_id)
    Date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_range =[]
    for i in range(len(Date_range)-1):
        start_date = pd.to_datetime(Date_range)[i].strftime("%Y-%m-%dT%H:%M:%S")
        end_date = pd.to_datetime(Date_range)[i+1].strftime("%Y-%m-%dT%H:%M:%S")
        date_range.append([])
        for j in range(1):
            date_range[i].append(start_date)
            date_range[i].append(end_date)
    example_data=pd.DataFrame()
    for i in range(len(date_range)):
        example_data=example_data.append(ek.get_news_headlines(f'{ticker_and_params}',
                                         count=10,
                                         date_from=date_range[i][0],
                                         date_to=date_range[i][1]),
                            ignore_index=False)
    news=example_data
    stories = []
    for i, storyId in enumerate(news['storyId']):
        try:
            html = ek.get_news_story(storyId)
            story = BeautifulSoup(html, 'html5lib').get_text()
            stories.append(story)
        except:
            stories.append('')
    news['story']=stories
    ## Sentiment Over Time
    sentiment = pd.DataFrame()
    sid = SentimentIntensityAnalyzer()
    for storyId in news['storyId']:
        row = news[news['storyId'] == storyId]
        scores = sid.polarity_scores(row['story'][0])
        sentiment = sentiment.append(pd.DataFrame(scores, index=[row['versionCreated'][0]]))
    sentiment.index = pd.DatetimeIndex(sentiment.index)
    sentiment.sort_index(inplace=True)
    sentiment_list=list(sentiment['compound'])
    news['sentiment']=sentiment_list
    #group by day
    dates_normal=[]
    for i in range(len(news['versionCreated'])):
        dates_normal.append(news['versionCreated'][i].strftime("%Y-%m-%d"))
    dates_normal_2=pd.to_datetime(dates_normal)
    news['dates_normal']=dates_normal_2
    
    #save to csv
    news.to_csv(f"{ticker}_news.csv")
    daily_sentiments_listed=news.groupby(pd.Grouper(key="dates_normal", freq="D"))['sentiment'].apply(list).reset_index()
    #Daily mood index
    DMI=[]
    for i in range(len(daily_sentiments_listed['sentiment'])):
        pos=0
        neg=0
        
        for j in range(len(daily_sentiments_listed['sentiment'][i])):
            try:
                if daily_sentiments_listed['sentiment'][i][j] > 0:
                    pos+=1
                elif daily_sentiments_listed['sentiment'][i][j] < 0:
                    neg+=1
            except:
                pass
        DMI.append(np.log((1+pos)/(1+neg)))
    daily_sentiments_listed['DMI']=DMI
    
    #Average Sentiment
    Average_S=[]
    for i in range(len(daily_sentiments_listed['sentiment'])):
        Average_S.append(np.mean(daily_sentiments_listed['sentiment'][i]))
    daily_sentiments_listed['Average_Sentiment']=pd.DataFrame(Average_S).fillna(0)
    
    #save sentiments to csv
    daily_sentiments_listed.to_csv(f"{ticker}_sentiment.csv", index=False)
    return daily_sentiments_listed, news