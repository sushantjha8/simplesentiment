
#! pip3 install feedparser
import feedparser as fp

import json
#! pip3 install newspaper3k
import newspaper
from newspaper import Article
from time import mktime
from datetime import datetime
import pandas as pd
import nltk as nlp
from nltk.tag.stanford import StanfordNERTagger as nert

import re, string, unicodedata
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
#from __future__ import print_function
from nltk.stem.snowball import SnowballStemmer
import itertools
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud
stemmer = SnowballStemmer("english")
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np



class uList:
	def __init__(self):
		self.total_limit = 1000
		self.data ={ }
	def flatten(self,list_to_flatten):
		for elem in list_to_flaten:
			if isinstance(elem,(list,tuple)):
				for x in flatten(elem):
					yield x
		else:
			yield elem
	def remove_non_ascii(self,word):
		new_words = []
		for word in words:
			new_word= word.lower()
			new_words.append(new_word)
		return new_words


	def to_lowercase(words):
		"""Convert all characters to lowercase from list of tokenized words"""
		new_words = []
		for word in words:
			new_word = word.lower()
			new_words.append(new_word)
		return new_words


	def remove_puctuation(self,words):
		new_words=[]
		for word in words:
			new_word =re.sub(r'[^\w\s]','',word)
			if new_word != '':
				new_words.append(new_word)
		return new_words
	def replace_number(self,words):
		p=inflect.engine()
		new_word=[]
		for word in words:
			if word.isdigit():
				new_word = p.number_to_words(word)
		return new_words
	def remove_stopwords(self,words):
		new_words=[]
		for word in words:
			if word not in stopwords.words('english'):
				new_words.append(word)
			return new_words
	def stem_words(self,words):
		stemmer = LancasterStemmer()
		stems = []
		for word in words:
			stem = stemmer.stem(word)
			stems.append(stem)
		return stems
	def lemmatize_verbs(self,words):
		lemmatizer = WordNetLemmatizer()
		lemmas = []
		for word in words:
			lemas=lemmatizer.lemmatize(word,pos='v')
			lemmas.append(lemma)
		return lemmas
	def normalize(self,words):
		words = remove_non_ascii(words)
		words = to_lowercase(words)
		words = remove_puctuation(words)
		words = replace_number(words)
		words = remove_stopwords(words)
		return ittertools.chain.from_iterable(words)

	def sent(self,uri,title,cl_text):
		from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
		nltk.download('vader_lexicon')
		sid = SIA()
		counter = 0
		sentiment =pd.DataFrame(columns=('Title','Sentance','Positive', 'Negative', 'Neutral', 'Compound'))
		sentiment12=[]
		for sentence1 in range(len(uri)):
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
			article_sentiment =pd.DataFrame([[title[sentence1],sentence,article_pos, article_neg, article_neu, article_compound]], columns=('Title','Sentance','Positive', 'Negative', 'Neutral', 'Compound'))
			sentiment = sentiment.append(article_sentiment, ignore_index = True)
		    
			
			sentiment12.append({
			'Title':title[sentence1],
			'Sentance':sentence,
			'Positive':article_pos, 
			'Negative':article_neg, 
			'Neutral':article_neu, 
			'Compound':article_compound
			})

		leng=sentiment["Title"]
		pos=[]
		neg=[]
		neu=[]
		com=[]
		tit=[]
		t_s=[]
		for i in range (len(leng)):
			tit.append(sentiment["Title"][i])
			pos.append(sentiment["Positive"][i])
			neg.append(sentiment["Negative"][i])
			neu.append(sentiment["Neutral"][i])
			com.append(sentiment["Compound"][i])
			avg_sentiment={
							"Title":tit,
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


	def newSentiment(self,uri):
		if uri != '':
			url=[]
			title=[]
			text=[]
			for info in range (len(uri)):
				a = Article(uri[info], language='en') # Chinese
				a.download()
				a.parse()

				di={"Url":uri[info],"Title":a.title,"Text":a.text}
				url.append(uri[info])
				title.append(a.title)
				text.append(a.text)
				self.data.update(di)
			news_article=pd.DataFrame()
			cl_text=[]
			t_Word=[]
			news_art=[]
			cloud=[]
			for i in range(len(uri)):
				cleanword=text[i]
				#print(title[i])
				cl_text.append(cleanword)
				#normalised=normalize(cleantext)
				tw = word_tokenize(text[i])
				cloud.append(tw)
				t_Word.append(tw)
				#news_article["tokenize sent"]=sent_tokenize(data["Text"])
				news_art.append(cleanword)

			sentiment12,avg_sentiment,avg_sent= self.sent(uri,title,cl_text)

			return sentiment12,avg_sentiment,avg_sent
		else:
			return "invalid url"

	



