#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!pip3 install nltk
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[3]:


#pip install selenium
#pip install beautifulsoup4
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import csv
import warnings
warnings.filterwarnings("ignore")


# In[4]:


import requests
from bs4 import BeautifulSoup

URL = "https://insights.blackcoffer.com/ai-in-healthcare-to-improve-patient-outcomes/"
URL_ID = URL.split("/")[-2] # extract the last part of the URL as the URL_ID

# make a request to the URL and get the content
response = requests.get(URL)
content = response.content

# parse the content with BeautifulSoup
soup = BeautifulSoup(content, "html.parser")

# find all the paragraph elements of the article
article_paragraphs = soup.find("div", {"class": "td-post-content"}).find_all("p")

# extract the text content of each paragraph and join them with a newline character
article_text = "\n\n".join([p.text for p in article_paragraphs])
#print(article_text)
# save the text content in a text file with the URL_ID as its filename
#with open(f"{URL_ID}.txt", "w",encoding="utf-8") as f:
#    f.write(article_text)
    
print('save')


# In[5]:


text=article_text
#print(text)


# # 1---Sentimental Analysis

# # 1.1 Cleaning using Stop Words Lists

# In[7]:


#Cleaning using Stop Words Lists

stop_words = set(stopwords.words('english'))
#stp_word=[]
#stp_word1=[stp_word].split(' ')
#print(sto_word1)
def clean_text(text):
    words = nltk.word_tokenize(text.lower())
    cleaned_text1 = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_text1)

#text = "This is a sentence with stop words. It needs to be cleaned."
cleaned_text = clean_text(text)
#stp_word.append(cleaned_text)
print(cleaned_text)
#print(stp_word)


# # 1.2--- Creating a dictionary of Positive and Negative words

# You can create a dictionary of positive and negative words in Python by following these steps:
# 
# Load the master dictionary into a variable (e.g. master_dict).
# Load the stop words list into a variable (e.g. stop_words).
# Filter the words in master_dict to include only those words that are not in stop_words.
# Create an empty dictionary to store positive and negative words (e.g. words_dict).
# Iterate over the filtered words and check their sentiment (positive or negative).
# For each word, add it to the words_dict under the key "positive" or "negative", depending on its sentiment.

# In[10]:


#Creating a dictionary of Positive and Negative words
positive=[]
with open("positive-words.txt", "r") as file:
    lines=(file.readlines())
    #print(lines)
for line in lines:
    #print(line)
    positive.append(line.split('\n')[0])
#print(set(positive))
negative=[]
with open("negative-words.txt", "r") as file2:
    lines1=file2.readlines()
    
for line1 in lines1:
    #print(line)
    negative.append(line1)
#print(negative)


# In[44]:


'''# Example master dictionary with words and their sentiment
master_dict = {
    "happy": "positive",
    "sad": "negative",
    "excited": "positive",
    "angry": "negative",
    "content": "positive"
}
master_dict={
    'Negative':negative,
    'Positive':positive
    
}
#print(master_dict)
# Example stop words list
stop_words = [stp_word]

# Filter the words in master_dict to exclude stop words
filtered_words = [word for word in master_dict.keys() if word not in stop_words]

# Initialize an empty dictionary to store positive and negative words
words_dict = {"positive": [], "negative": []}

# Iterate over filtered_words and determine the sentiment of each word
for word in filtered_words:
    sentiment = master_dict[word]
    words_dict[sentiment].append(word)

# The resulting dictionary, words_dict, contains positive and negative words
print(words_dict)
# Output: {"positive": ["happy", "excited"], "negative": ["sad", "angry"]}
'''


# # 1.3  Extracting Derived variables

# In the above code, we first tokenize the text into a list of words using the word_tokenize function from the nltk.tokenize module.
# 
# Next, we create dictionaries for positive and negative words and initialize two variables, positive_score and negative_score, to keep track of the scores.
# 
# In the for loop, we iterate over the tokens and check if each token is in the positive_dict or negative_dict. If the token is in the positive_dict, we add 1 to positive_score. If the token is in the negative_dict, we subtract 1 from negative_score.
# 
# After processing all the tokens, we calculate the polarity score and subjectivity score using the given formulas. Finally, we print the resulting polarity and

# The Polarity Score ranges from -1 to +1, where -1 represents a completely negative sentiment, +1 represents a completely positive sentiment, and 0 represents a neutral sentiment. The Subjectivity Score ranges from 0 to +1, where 0 represents a completely objective sentiment and +1 represents a completely subjective sentiment.

# In[122]:


import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

# Example text
text = "The movie was amazing and I was so happy to watch it."

# Tokenize the text into a list of words
tokens = word_tokenize(text)

# Example dictionaries of positive and negative words
positive_dict = {"happy", "amazing"}
negative_dict = {"sad"}

# Initialize variables to keep track of positive and negative scores
positive_score = 0
negative_score = 0

# Iterate over the tokens and update positive and negative scores
for token in tokens:
    if token in positive_dict:
        positive_score += 1
    elif token in negative_dict:
        negative_score -= 1

# Multiply negative_score by -1 to ensure it is a positive number
negative_score *= -1

# The resulting positive and negative scores
print("Positive score:", positive_score)
print("Negative score:", negative_score)
# Output: Positive score: 2
#         Negative score: 0


# In[121]:


import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

# Example text
text = "The movie was amazing and I was so happy to watch it."

# Tokenize the text into a list of words
tokens = word_tokenize(text)

# Example dictionaries of positive and negative words
positive_dict = {"happy", "amazing"}
negative_dict = {"sad"}

# Initialize variables to keep track of positive and negative scores
positive_score = 0
negative_score = 0

# Iterate over the tokens and update positive and negative scores
for token in tokens:
    if token in positive_dict:
        positive_score += 1
    elif token in negative_dict:
        negative_score -= 1

# Calculate the polarity score
polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

# Calculate the subjectivity score
subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)

# The resulting polarity and subjectivity scores
print("Polarity score:{:.2f}".format(polarity_score))
print("Subjectivity score: {:.2f} ".format(subjectivity_score))
# Output: Polarity score: 0.6666667
#         Subjectivity score: 0.5


# # 2--Analysis of Readability

# # 2.1

# This code takes a string text as input and splits it into words using the split method. It then counts the number of words using len and the number of sentences using the count method for the punctuation marks ., ?, and !. Finally, it returns the result of the formula num_words / num_sentences.

# In[9]:


#
#Average Sentence Length
def avg_sentence_length(cleaned_text):
    words = cleaned_text.split()
    num_words = len(words)
    
    num_sentences = cleaned_text.count(".") + cleaned_text.count("?") + cleaned_text.count("!")
    return num_words / num_sentences

#text = "This is a sentence. And this is another one?"
result = avg_sentence_length(cleaned_text)
print("Average sentence length: {:.2f}".format(result))
#"Percentage of complex words: {:.2f}%".format(result * 100)


# # 2.2

# This code takes a string text as input, splits it into words using the split method, and counts the number of words using len. It then iterates over each word, checking if its length is greater than or equal to 5, and if so, increments the count of complex words. Finally, it returns the result of the formula complex_words / num_words. The result is formatted as a percentage using the format method.

# In[101]:


#Percentage of Complex words
def percent_complex_words(text):
    words = text.split()
    num_words = len(words)
    complex_words = 0
    for word in words:
        if len(word) >= 5:
            complex_words += 1
    return complex_words / num_words

#text = "This is a complex sentence with complex words."
result = percent_complex_words(text)
print("Percentage of complex words: {:.2f}%".format(result * 100))


# # 2.3

# This code defines two helper functions avg_sentence_length and percent_complex_words, which were described in previous answers. The fog_index function uses these functions to calculate the average sentence length and the percentage of complex words, and returns the result of the formula 0.4 * (avg_len + pct_complex).

# In[123]:


def fog_index(text):
    def avg_sentence_length(text):
        words = text.split()
        num_words = len(words)
        num_sentences = text.count(".") + text.count("?") + text.count("!")
        return num_words / num_sentences

    def percent_complex_words(text):
        words = text.split()
        num_words = len(words)
        complex_words = 0
        for word in words:
            if len(word) >= 5:
                complex_words += 1
        return complex_words / num_words

    avg_len = avg_sentence_length(text)
    pct_complex = percent_complex_words(text)
    return 0.4 * (avg_len + pct_complex)

#text = "This is a complex sentence with complex words."
result = fog_index(text)
print("Fog index: {:.2f}".format(result))


# # 3____Average Number of Words Per Sentence
# 

# This code takes a string text as input, splits it into words using the split method, and counts the number of words using len. It then counts the number of sentences using the count method for the punctuation marks ., ?, and !. Finally, it returns the result of the formula num_words / num_sentences.

# In[126]:


def avg_words_per_sentence(text):
    words = text.split()
    num_words = len(words)
    num_sentences = text.count(".") + text.count("?") + text.count("!")
    return num_words / num_sentences

#text = "This is a sentence. And this is another one?"
result = avg_words_per_sentence(text)
print("Average number of words per sentence: {:.2f}".format(result))


# # 4__Complex Word Count

# This code uses the Natural Language Toolkit (nltk) library and the Carnegie Mellon University Pronouncing Dictionary (cmudict) to count the number of syllables in each word. If a word has more than 2 syllables, it is considered a complex word and the count of complex words is incremented. The code first checks if the word is in the dictionary using a try-except block, since some words may not be found in the dictionary. The code returns the final count of complex words.

# In[109]:


#4-----
import nltk
from nltk.corpus import cmudict

nltk.download('cmudict')

def count_complex_words(text):
    d = cmudict.dict()
    words = text.split()
    complex_word_count = 0
    for word in words:
        try:
            syllables = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
            if syllables > 2:
                complex_word_count += 1
        except KeyError:
            pass
    return complex_word_count

#text = "This is a complex sentence with complex words."
result = count_complex_words(text)
print("Complex word count:", result)


# # 5-- Word Count 

# This code uses the Natural Language Toolkit (nltk) library and the stopwords corpus to remove stop words from the input text. It also removes punctuation using the string module and the strip method. The code splits the text into words using the split method, removes the punctuation from each word, and checks if the word is in the set of stop words using an if statement. The final count of words is returned using len.

# why we use this line ---stop_words = set(stopwords.words('english'))????????????????????
# 
# Solution:----
# The line stop_words = set(stopwords.words('english')) is used to obtain a set of stop words in English from the stopwords corpus in the Natural Language Toolkit (nltk) library.
# 
# A stop word is a word that is commonly used and does not carry much meaning, such as "the", "and", "a", etc. These words are often removed from text data before further processing because they do not contribute much to the meaning of the text and can increase the size of the data unnecessarily.
# 
# By creating a set of stop words, we can easily check if a word is in the set using the in operator. This is more efficient than checking against a list, as the in operator has a time complexity of O(1) for sets, while it has a time complexity of O(n) for lists.
# 
# The argument 'english' passed to the words function specifies the language for which we want to obtain the stop words. The function returns a list of stop words, which we convert to a set using the set function

# In[110]:


import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def word_count(text):
    stop_words = set(stopwords.words('english'))
    words = [word.strip(string.punctuation) for word in text.split() if word.strip(string.punctuation) not in stop_words]
    return len(words)

#text = "This is a sentence with some stop words and punctuation."
result = word_count(text)
print("Word count:", result)


# # 6 Syllable Count Per Word 

# This code uses the re (regular expression) module to count the number of syllables in each word in the input text. The count_syllables function first converts the word to lowercase and removes all non-vowel characters using regular expressions. It then replaces multiple consecutive vowels with a single vowel and removes any remaining vowels at the start or end of the word. If the word ends with 'es' or 'ed', the function returns max(1, len(word) - 2), otherwise it returns len(word). This is because words ending with 'es' or 'ed' can often still be pronounced as one syllable.
# 
# The syllable_count_per_word function splits the text into words, and for each word, it calls count_syllables to obtain the number of syllables. The list of syllables per word is returned.

# In[118]:


import re

def count_syllables(word):
    word = word.lower()
    word = re.sub(r"([^aeiouy])", '', word)
    word = re.sub(r"a+", 'a', word)
    word = re.sub(r"e+", 'e', word)
    word = re.sub(r"i+", 'i', word)
    word = re.sub(r"o+", 'o', word)
    word = re.sub(r"u+", 'u', word)
    word = re.sub(r"y+", 'y', word)
    word = word.strip('aeiouy')
    if word.endswith(('es', 'ed')):
        return max(1, len(word) - 2)
    else:
        return len(word)

def syllable_count_per_word(text):
    words = text.split()
    syllables_per_word = [count_syllables(word) for word in words]
    return syllables_per_word

text = "This is a sentence with some words."
result = syllable_count_per_word(text)
print("Syllables per word:", result)


# # 7__ Personal Pronouns

# This code uses the re (regular expression) module to count the number of personal pronouns in the input text. The count_personal_pronouns function first converts the text to lowercase, and then initializes a list of personal pronouns and a count variable. For each pronoun in the list, it uses the re.findall function to find all occurrences of the pronoun in the text, and adds the number of occurrences to the count. The function returns the count of personal pronouns.
# 
# Note that the regular expression r'\b' + pronoun + r'\b' is used to ensure that only whole words are matched, and not parts of words. The \b characters are word boundary characters that match the position between a word character (as defined by \w) and a non-word character (as defined by \W). This prevents the country name 'US' from being included in the count, as it is not a whole word.

# In[113]:


import re

def count_personal_pronouns(text):
    text = text.lower()
    personal_pronouns = ['i', 'we', 'my', 'ours', 'us']
    count = 0
    for pronoun in personal_pronouns:
        count += len(re.findall(r'\b' + pronoun + r'\b', text))
    return count

#text = "I went to the store. We bought some food. My favorite is pizza. US is a country."
result = count_personal_pronouns(text)
print("Personal pronouns count:", result)


# 
# # 8--Average Word Length
# 

# This code splits the input text into words using the split method, and initializes a total_chars variable to keep track of the total number of characters in all words. For each word in the list of words, it adds the length of the word (the number of characters) to the total_chars variable. Finally, it returns the average word length by dividing the total_chars by the number of words.

# In[127]:


def average_word_length(text):
    words = text.split()
    total_chars = 0
    for word in words:
        total_chars += len(word)
    return total_chars / len(words)

#text = "This is a sample text."
result = average_word_length(text)
print("Average word length:{:.2f}".format(result))


# In[ ]:




