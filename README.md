# Simplesentiment
[![Build Status](https://travis-ci.org/sushantjha8/simplesentiment.svg?branch=master)](https://travis-ci.org/sushantjha8/simplesentiment)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-team/keras/blob/master/LICENSE)

## You have just found SimpleSentiment.

SimpleSentiment is a high-level sentiment analysis API, written in Python and capable of running on top of [NLTK](https://github.com/nltk/nltk). It was developed with a focus on enabling fast experimentation on diffrent sentiment analysis technique.

Use simpleSentiment if you need a deep learning sentiment analysis on any text.:

- Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
- Supports both convolutional networks and recurrent networks, as well as combinations of the two.
- Runs seamlessly on CPU and GPU.

SimpleSentient is compatible with: __Python 3.6__.


------------------


## Guiding principles

- __User friendliness.__ SimpleSentient is an API designed for minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.



- __Easy extensibility.__ A model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as little restrictions as possible. In cleaning of text,sentient detail analysis inormation, are all standalone modules that you can combine to create new models.



------------------


## Getting started: 30 seconds to simplesentiment

Here is the `url content analysis` doing sentiment analysis on multiple blogs :

```python
from simplesentiment.uriList import uList

```

Getting the sentiment `.newSentiement()`:

```python
multiblog_sentient=uList()

detail_Analysis,analysis_o,avg_sentiment=multiblog_sentient.newSentiement(['http://blogurl'])
```
newSentiement
##Input
uri=Input taken list of list of blogs and uri
##Output
It returns 3 level of sentient analysis
```
detail_Analysis,analysis_o,avg_sentiment
```
##Installation
For Python3.6
```
 pip install -i https://test.pypi.org/simple/ simplesentiment
                    or
 pip3 install -i https://test.pypi.org/simple/ simplesentiment
  ```
