from __future__ import annotations

import re

import nltk
from nltk import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.stem import SnowballStemmer 
from nltk.stem import PorterStemmer 
from nltk.corpus import wordnet

import numpy as np
import pandas as pd
from IPython.display import display

downloaded = nltk.download('stopwords')
downloaded = nltk.download('averaged_perceptron_tagger')
downloaded = nltk.download('punkt')
downloaded = nltk.download('wordnet')

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import logging

class NLTKNormalizer(BaseEstimator, TransformerMixin):
	""" Class to normalize tokens using NLTK Stemmers and Lemmatizers """
	def __init__(self, 
				 normalizer:str|None = None, # the normalizer to use None, 'PorterStemmer', 'SnowballStemmer', 'WordNetLemmatizer'
				 lowercase:bool = True, # whether to lowercase the tokens before normalizing
				 stop_word_list:str|None = None, # the stop word list to use - None, 'sklearn', 'nltk', 'both'
				 extra_stop_words:list = [] # extra stop words to add to the stop word list
				 ):
		self.normalizer_type = normalizer
		if self.normalizer_type == 'PorterStemmer':
			self.normalizer = PorterStemmer()
		elif self.normalizer_type == 'SnowballStemmer':
			self.normalizer = SnowballStemmer('english')
		elif self.normalizer_type == 'WordNetLemmatizer':
			self.normalizer = WordNetLemmatizer()
		elif self.normalizer_type is None:
			self.normalizer = None
		else:
			logging.warning(f'Unknown normalizer_type - ignoring, no normalizer will be used')
			self.normalizer_type = None
			self.normalizer = None

		self.lowercase = lowercase

		self.stop_word_lists = get_stopword_lists()

		self.stop_word_list = stop_word_list
		if self.stop_word_list in ['sklearn', 'nltk', 'both']:
			self.stop_words = self.stop_word_lists[self.stop_word_list]
		else:
			self.stop_word_list = None
			self.stop_words = []

		self.extra_stop_words = extra_stop_words

		if len(extra_stop_words) > 0:
			self.stop_words += extra_stop_words

	def fit(self, X, y=None):
		return self

	def transform(self, 
				  X):
		return [self.normalize(tokens) for tokens in X]

	def normalize(self,
				  tokens: list[str] # the list of tokens to normalize
				  ) -> list[str]: # the normalized tokens

		# if using a normalizer then iterate through tokens and return the normalized tokens ...
		if self.normalizer_type == 'PorterStemmer':
			tokens = [self.normalizer.stem(t) for t in tokens]
		elif self.normalizer_type == 'SnowballStemmer':
			tokens = [self.normalizer.stem(t) for t in tokens]
		elif self.normalizer_type == 'WordNetLemmatizer':
			# NLTK's lemmatiser needs parts of speech, otherwise assumes everything is a noun
			pos_tokens = nltk.pos_tag(tokens)
			lemmatised_tokens = []
			for token in pos_tokens:
				# NLTK's lemmatiser needs specific values for pos tags - this rewrites them ...
				# default to noun
				tag = wordnet.NOUN
				if token[1].startswith('J'):
					tag = wordnet.ADJ
				elif token[1].startswith('V'):
					tag = wordnet.VERB
				elif token[1].startswith('R'):
					tag = wordnet.ADV
				lemmatised_tokens.append(self.lemmatize(token[0],tag))
			tokens = lemmatised_tokens
		else:
			# no normaliser so just return tokens
			tokens = tokens

		# lowercase the tokens if required
		if self.lowercase:
			tokens = [t.lower() for t in tokens]
		
		# removing stop words if required
		if self.stop_words is not None:
			tokens = [t for t in tokens if t not in self.stop_words]
		
		return tokens

class NLTKTokenizer(BaseEstimator, TransformerMixin):
	""" Class to tokenize text using NLTK tokenizers """
	def __init__(self, tokenizer:str = 'word_tokenize'):
		if tokenizer in ['sklearn', 'word_tokenize', 'wordpunct']:
			self.tokenizer = tokenizer
		else:
			logging.warning(f'Unknown tokenizer - defaulting to sklearn')
			self.tokenizer = 'sklearn'

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		tokens = []
		for text in X:
			# add more tokenizers here if needed ...
			if self.tokenizer == 'sklearn':
				tokenizer = RegexpTokenizer(r"(?u)\b\w\w+\b") # this is copied straight from sklearn source
				tokens.append(tokenizer.tokenize(text))
			elif self.tokenizer == 'wordpunct':
				tokens.append(wordpunct_tokenize(text))
			else:
				tokens.append(word_tokenize(text))
		return tokens
	
def pass_tokens(tokens):
	return tokens

def get_stopword_lists():
	nltk_stop_words = nltk.corpus.stopwords.words('english')

	stop_word_lists = {
		'sklearn': list(sklearn_stop_words),
		'nltk': nltk_stop_words,
		'both': list(set(nltk_stop_words).union(set(sklearn_stop_words))),
	}
	return stop_word_lists


def get_preview(docs, targets, target_names, doc_id, max_len=0):
	""" Get a nice preview of a document """
	preview = ''
	if max_len < 1:
		preview += 'Label\n'
		preview += '=====\n'
	else:
		preview += str(doc_id)
		preview += '\t'
	preview += target_names[targets[doc_id]]
	if max_len < 1:
		preview += '\n\nFull Text\n'
		preview += '=========\n'
		if docs[doc_id].strip() == '':
			preview += 'No text in this document.'
		else:
			preview += docs[doc_id]
		preview += '\n'
	else:
		excerpt = get_excerpt(docs[doc_id], max_len)
		preview += '\t' + excerpt
	return preview

# regular expression to combine whitespace
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

def get_excerpt(text, max_len):
	""" Get an excerpt of a document """
	excerpt = _RE_COMBINE_WHITESPACE.sub(' ',text[0:max_len])
	if max_len < len(text):
		excerpt += '...'
	return excerpt.strip()


def nb_binary_display_most_informative_features(pipeline, dataset, features_to_show=20):
	vect = pipeline.named_steps['vectorizer']
	clf = pipeline.named_steps['classifier']
	feature_names = vect.get_feature_names_out()
	logodds=clf.feature_log_prob_[1]-clf.feature_log_prob_[0]

	df = pd.DataFrame({
		'Feature': feature_names,
		'Log-Odds': logodds,
	})

	print("Features most indicative of",dataset.target_names[0])
	print('============================' + '='*len(dataset.target_names[0]))

	sorted_df = df.sort_values('Log-Odds', ascending=True).head(features_to_show)
	display(sorted_df)

	print("Features most indicative of",dataset.target_names[1])
	print('============================' + '='*len(dataset.target_names[1]))

	sorted_df = df.sort_values('Log-Odds', ascending=False).head(features_to_show)
	display(sorted_df)

def get_feature_frequencies(pipeline, text):
	preprocessor = Pipeline(pipeline.steps[:-1])
	frequency = preprocessor.transform([text]).toarray()[0].T
	df = pd.DataFrame(frequency, index=preprocessor.named_steps['vectorizer'].get_feature_names_out(), columns=['frequency'])
	df = df[df['frequency'] > 0].sort_values('frequency', ascending=False)
	if len(df) < 1:
		return 'No features extracted from this document.'
	else:
		return df

__all__ = ['NLTKNormalizer', 'NLTKTokenizer', 'pass_tokens', 'get_stopword_lists', 'get_excerpt', 'get_preview', 'nb_binary_display_most_informative_features', 'get_feature_frequencies']
