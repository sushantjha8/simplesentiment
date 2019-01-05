from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import nltk
import pandas as pd
import json
from nltk.stem.snowball import SnowballStemmer
import itertools
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud

import nltk
import numpy as np

class sentanceanalyser:
	"""docstring for ClassName"""
	def __init__(self):
		pass
	def sentance_sentiment(self,text):
		nltk.download('vader_lexicon')
		cl_text=tokenize.sent_tokenize(text)

		sid = SentimentIntensityAnalyzer()
		counter = 0
		sentiment =pd.DataFrame(columns=('Sentance','Positive', 'Negative', 'Neutral', 'Compound'))
		sentiment12=[]
		for sentence1 in range(len(cl_text)):
			sentence=cl_text[sentence1]
			article_compound = 0
			article_neg = 0

			article_pos = 0
			article_neu = 0
			counter = counter + 1

			ss = sid.polarity_scores(sentence)
			article_compound = article_compound + ss['compound']
			article_compound=round(article_compound,2)
			article_neg = article_neg + ss['neg']
			article_neg=round(article_neg,2)
			article_pos = article_pos + ss['pos']
			article_pos=round(article_pos,2)
			article_neu = article_neu + ss['neu']
			article_neu=round(article_neg,2)
			article_sentiment =pd.DataFrame([[sentence,article_pos, article_neg, article_neu, article_compound]], columns=('Sentance','Positive', 'Negative', 'Neutral', 'Compound'))
			sentiment = sentiment.append(article_sentiment, ignore_index = True)
		    
			
			sentiment12.append({
			'Sentance':sentence,
			'Positive':article_pos, 
			'Negative':article_neg, 
			'Neutral':article_neu, 
			'Compound':article_compound
			})

		leng=sentiment["Sentance"]
		pos=[]
		neg=[]
		neu=[]
		com=[]
		tit=[]
		t_s=[]
		for i in range (len(leng)):
			pos.append(sentiment["Positive"][i])
			neg.append(sentiment["Negative"][i])
			neu.append(sentiment["Neutral"][i])
			com.append(sentiment["Compound"][i])
			avg_sentiment={
							"Positive":pos,
							"Negative":neg,
							"Neutral":neu,
							"Compound":com	
			}
			t_s.append(avg_sentiment)
		t_pos=sum(pos)
		t_ne=sum(neg)
		t_neu=sum(neu)
		t_com=sum(com)
		avg_sent=[{"Total Positive":round(t_pos,2),"Total_Negative":round(t_ne,2),"Total_Neutral":round(t_neu,2),"Total_Compound":round(t_com,2)}]
		return sentiment12,avg_sentiment,avg_sent
