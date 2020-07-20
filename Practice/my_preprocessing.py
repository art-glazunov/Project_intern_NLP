
"""
My little collection of functions for preprocessing for a single text


List of functions:

1. emoji_replacer(text,emoji_list,replacers)
- Transforms emoji into words

2. lemmatize_lower_case(text)
- Lemmatizes the text, transforming it to the lower case previously

   lemmatize_lower_case2(text):
  the same, but with pymystem  

3. delete_stop_words(text)
- Deletes Russian stopwords 

4. numbers_to_text(text)
- Converts numbers into Russian text

5. def ne_loc_extraction(text,
                  segmenter,morph_vocab,
                  morph_tagger,ner_tagger,
                  del_names=False,del_addr=False)
- Extracts, normalizes and deletes (optional) named entities and locations
   
6. add_normal_ne_and_loc(text,normal_ne_loc,add_names = True add_only_first_names = False,
                 add_only_last_names = False, add_locations = True):
- Adds extracted and normalized named entities and locations in the end

7. delete_digits(text):
- Deletes digits


"""
import re

from bs4 import BeautifulSoup
import pymorphy2
from num2words import num2words

from nltk.corpus import stopwords


from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,

    NamesExtractor,
    AddrExtractor,
    Doc
)

import numpy as np

from pymystem3 import Mystem


#Initialise Natasha main tools

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)
addr_extractor = AddrExtractor(morph_vocab)


lemmatizer = Mystem()


def emoji_replacer(text,emoji_list,replacers):
  #Transform emoji into words

  for index,emoji in enumerate(emoji_list):
    text = text.replace(emoji,' '+ replacers[index] +' ')

  return text



def text_early_preproc(text,del_html = True,del_punct_sp_chars=True,
                 del_underscore=True, del_digits=False):
  #Clean the text from artifacts and punctuation

  #Delete whitespaces and special string symbols
  text = re.sub("^\s+|\n|\r|\s+$", ' ', text)

  #Delete html tags
  if del_html:
    soap = BeautifulSoup(text, 'html.parser')
    text = soap.get_text()

  #Delete punctuation and other artifacts
  if del_punct_sp_chars:
    text = re.sub(r'[^\w\s]','',text)

  #Delete '_'
  if del_underscore:
    text = text.replace('_','')

  #Delete digits
  if del_digits:
    text = re.sub(r'\d+', '', text)

                 
  return text


def lemmatize_lower_case(text):
  #lemmatizationa in lower case

  words = text.lower().split()
  morph = pymorphy2.MorphAnalyzer()

  normal_tokens= [morph.parse(word)[0].normal_form for word in words]

  return " ".join(normal_tokens)


def lemmatize_lower_case2(text):
  #lemmatizationa in lower case using pymystem

  lemmas = lemmatizer.lemmatize(text.lower())


  return " ".join(lemmas)




def delete_stop_words(text):
  #delete Russian stopwords

  tokens = [token for token in text.split() if token not in stopwords.words("russian")]
  text = " ".join(tokens)
  return text

def numbers_to_text(text):
  #Converts numbers into Russian text

  tokens = text.split()
  text = " ".join([num2words(token,lang='ru') if token.isnumeric() else token for token in tokens ])
  
  return text

def delete_digits(text):
  #Deletes digits
  text = re.sub(r'\d+', '', text)

  return text


def ne_loc_extraction(text,del_names=False,del_addr=False):      
                 
  #Extract, normalize and delete (optional) named entities and locations


  doc = Doc(text)

  doc.segment(segmenter)
  doc.tag_ner(ner_tagger)
  doc.tag_morph(morph_tagger)


  for span in doc.spans:
    span.normalize(morph_vocab)

  for span in doc.spans:

    if span.type == 'PER':
      span.extract_fact(names_extractor)

    if span.type == 'LOC':
      span.extract_fact(addr_extractor)

  if del_names:
    for span in doc.spans:
      if span.type == 'PER':
        text = text.replace(span.text,'')


  if del_addr:
    for span in doc.spans:
      if span.type == 'LOC':
        text = text.replace(span.text,'')

  normal_ne_loc = {}
  normal_ne_loc['NAMES'] = [span.normal for span in doc.spans if span.type == 'PER']
  normal_ne_loc['LOCATIONS'] = [span.normal for span in doc.spans if span.type == 'LOC']

  return text, normal_ne_loc


def add_normal_ne_and_loc(text,normal_ne_loc,add_names = True, add_only_first_names = False,
                 add_only_last_names = False, add_locations = True):
  #Add extracted and normalized named entities and locations in the end
  
  names = []
  locations = []
  
  if add_names:  
    if add_only_first_names:      

      names = [ne.split()[0] for ne in normal_ne_loc['NAMES']]

    elif add_only_last_names:   
       
      names = [ne.split()[len(ne.split())-1] for ne in normal_ne_loc['NAMES']]
  
  else:
    names = ["_".join(ne.split()) for ne in normal_ne_loc['NAMES']]
  
  
  if add_locations:
  	locations = ["_".join(ne.split()) for ne in normal_ne_loc['LOCATIONS']]

  return " ".join([text] + names + locations)


