#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import nltk
from nltk.corpus import cmudict
nltk.download('cmudict')
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore")

# Load the Excel file into a pandas DataFrame
df = pd.read_excel("Blackcoffer_Input.xlsx")

# Extract the URL column into a list
url_list = df["URL"].tolist()
'''for i in range(5):
    lst=url_list[i]
    print(lst)
'''   
lst=[url_list[x] for x in range(3)]
print(lst) 


# #  belowe code use only for save  extract data to filenames  

# In[62]:


import requests
from bs4 import BeautifulSoup
import csv


#csv_file=open('blackcoffer_output.csv', 'w')
#csv_writer=csv.writer(csv_file)
#csv_writer.writerow(['URL','POSITIVE SCORE','NEGATIVE SCORE','POLARITY SCORE','SUBJECTIVITY SCORE','AVG SENTENCE LENGTH','PERCENTAGE OF COMPLEX WORDS','FOG INDEX','AVG NUMBER OF WORDS PER SENTENCE','COMPLEX WORD COUNT','WORD COUNT','SYLLABLE PER WORD','PERSONAL PRONOUNS','AVG WORD LENGTH'])
count=0
for i in lst:
    URL=str(i)
    print(URL)
    #URL = "https://insights.blackcoffer.com/ai-in-healthcare-to-improve-patient-outcomes/"
    URL_ID = URL.split("/")[-2] # extract the last part of the URL as the URL_ID
    print(URL_ID)
    # make a request to the URL and get the content
    response = requests.get(URL)
    content = response.content

    # parse the content with BeautifulSoup
    soup = BeautifulSoup(content, "html.parser")

    # find all the paragraph elements of the article
    article_paragraphs = soup.find("div", {"class": "td-post-content"}).find_all("p")

    # extract the text content of each paragraph and join them with a newline character
    text = "\n\n".join([p.text for p in article_paragraphs])
    
    
    # save the text content in a text file with the URL_ID as its filename
    #with open(f"{URL_ID}.txt", "w",encoding="utf-8") as f:
    #    f.write(text)
    print('-----------------------------------------')
    print('-----------------------------------------')
    
print('save')


# In[ ]:


#Creating a dictionary of Positive and Negative words
positive=[]
with open("positive-words.txt", "r") as file:
    lines=(file.readlines())
    #print(lines)
for line in lines:
    #print(line)
    positive.append(line.split('\n')[0])
positive_dict=set(positive)

negative=[]
with open("negative-words.txt", "r") as file2:
    lines1=file2.readlines()   
for line1 in lines1:
    #print(line)
    negative.append(line1.split('\n')[0])
#print(negative)
negative_dict = set(negative)


# In[ ]:


#Creating a dictionary of stop word
StopWords_Names=[]
with open("StopWords_Names.txt", "r") as file:
    lines=(file.readlines())
    #print(lines)
for line in lines:
    #print(line)
    StopWords_Names.append(line.split('\n')[0])
#print(set(positive))

StopWords_Geographic=[]
with open("StopWords_Geographic.txt", "r") as file2:
    lines1=file2.readlines()
for line1 in lines1:
    #print(line)
    StopWords_Geographic.append(line1)
#print(StopWords_Geographic)

StopWords_GenericLong=[]
with open("StopWords_GenericLong.txt", "r") as file2:
    lines1=file2.readlines()
for line1 in lines1:
    #print(line)
    StopWords_GenericLong.append(line1)
#print(StopWords_GenericLong)

StopWords_Generic=[]
with open("StopWords_Generic.txt", "r") as file:
    lines=(file.readlines())
    #print(lines)
for line in lines:
    #print(line)
    StopWords_Generic.append(line.split('\n')[0])
#print(StopWords_Generic)

StopWords_DatesandNumbers=[]
with open("StopWords_DatesandNumbers.txt", "r") as file2:
    lines1=file2.readlines()
for line1 in lines1:
    #print(line)
    StopWords_DatesandNumbers.append(line1)
#print(StopWords_DatesandNumbers.txt)
StopWords_Currencies=[]
with open("StopWords_Currencies.txt", "r") as file:
    lines=(file.readlines())
    #print(lines)
for line in lines:
    #print(line)
    StopWords_Currencies.append(line.split('\n')[0])
#print(StopWords_Currencies)

StopWords_Auditor=[]
with open("StopWords_Auditor.txt", "r") as file:
    lines=(file.readlines())
    #print(lines)
for line in lines:
    #print(line)
    StopWords_Auditor.append(line.split('\n')[0])
#print(StopWords_Auditor)


# In[ ]:


#Cleaning using Stop Words Lists

stop_words = set(stopwords.words('StopWords_Auditor'))
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


# In[67]:


import re
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

#csv_file=open('blackcoffer_output.csv', 'w')
#csv_writer=csv.writer(csv_file)
#csv_writer.writerow(['URL','POSITIVE SCORE','NEGATIVE SCORE','POLARITY SCORE','SUBJECTIVITY SCORE','AVG SENTENCE LENGTH','PERCENTAGE OF COMPLEX WORDS','FOG INDEX','AVG NUMBER OF WORDS PER SENTENCE','COMPLEX WORD COUNT','WORD COUNT','SYLLABLE PER WORD','PERSONAL PRONOUNS','AVG WORD LENGTH'])
#csv_writer.writerow(['URL','Average sentence length', 'Percentage of complex words', 'Fog index','Average number of words per sentence', 'Complex word count', 'Word count','Personal pronouns count', 'Average word length'])
results = []
for i in lst:
    URL=str(i)
    #URL_ID = URL.split("/")[-2] # extract the last part of the URL as the URL_ID
    # make a request to the URL and get the content
    response = requests.get(URL)
    content = response.content
    # parse the content with BeautifulSoup
    soup = BeautifulSoup(content, "html.parser")
    # find all the paragraph elements of the article
    article_paragraphs = soup.find("div", {"class": "td-post-content"}).find_all("p")
    # extract the text content of each paragraph and join them with a newline character
    text = "\n".join([p.text for p in article_paragraphs])
    
    
    
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize
    # Tokenize the text into a list of words
    tokens = word_tokenize(text)
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
    print("Positive score:{:.2f}".format(positive_score))
    print("Negative score:{:.2f}".format(negative_score))
   
    
    
    # Calculate the polarity score
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    # Calculate the subjectivity score
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)
    # The resulting polarity and subjectivity scores
    print("Polarity score: {:.2f}".format(polarity_score))
    print("Subjectivity score: {:.2f}".format(subjectivity_score))

    
    
    
    #Average Sentence Length
    def avg_sentence_length(text):
        words = text.split()
        num_words = len(words)
        num_sentences = text.count(".") + text.count("?") + text.count("!")
        return num_words / num_sentences
    result1 = avg_sentence_length(text)
    print("Average sentence length: {:.2f}".format(result1))
    
    
    #Percentage of Complex words
    def percent_complex_words(text):
        words = text.split()
        num_words = len(words)
        complex_words = 0
        for word in words:
            if len(word) >= 5:
                complex_words += 1
        return complex_words / num_words
    result2 = percent_complex_words(text)
    print("Percentage of complex words: {:.2f}%".format(result2 * 100))

    

    #Fog Index
    avg_len = avg_sentence_length(text)
    pct_complex = percent_complex_words(text)
    result3=0.4 * (avg_len + pct_complex)
    print("Fog index:{:.2f}".format(result3))
 


    def avg_words_per_sentence(text):
        words = text.split()
        num_words = len(words)
        num_sentences = text.count(".") + text.count("?") + text.count("!")
        return num_words / num_sentences
    result4 = avg_words_per_sentence(text)
    print("Average number of words per sentence:{:.2f}".format(result4))
    

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
    result5 = count_complex_words(text)
    print("Complex word count:", result5)
    
    
    def word_count(text):
        stop_words = set(stopwords.words('english'))
        words = [word.strip(string.punctuation) for word in text.split() if word.strip(string.punctuation) not in stop_words]
        return len(words)
    result6 = word_count(text)
    print("Word count:", result6)
    
    #Syllable Count Per Word
    count = 0
    for word in text:
        vowels = "aeiouAEIOU"
        for letter in word:
            if letter in (vowels or ['es', 'ed']) :
                count += 1
    print("Syllable Count Per Word",count)
   

    def count_personal_pronouns(text):
        text = text.lower()
        personal_pronouns = ['i', 'we', 'my', 'ours', 'us']
        count = 0
        for pronoun in personal_pronouns:
            count += len(re.findall(r'\b' + pronoun + r'\b', text))
        return count
    result7 = count_personal_pronouns(text)
    print("Personal pronouns count:", result7)
    

    def average_word_length(text):
        words = text.split()
        total_chars = 0
        for word in words:
            total_chars += len(word)
        return total_chars / len(words)
    result8 = average_word_length(text)
    print("Average word length:{:.2f}".format(result8))
    print('-----------------------------------------')
    print('-----------------------------------------')
     
    # Append the results for each iteration to the list
    results.append([URL,positive_score,negative_score,polarity_score,subjectivity_score,result1, result2, result3, result4, result5, result6, count, result7, result8])
print('Save')
# Create a DataFrame from the results list
df= pd.DataFrame(results, columns=(['URL','POSITIVE SCORE','NEGATIVE SCORE','POLARITY SCORE','SUBJECTIVITY SCORE','AVG SENTENCE LENGTH','PERCENTAGE OF COMPLEX WORDS','FOG INDEX','AVG NUMBER OF WORDS PER SENTENCE','COMPLEX WORD COUNT','WORD COUNT','SYLLABLE PER WORD','PERSONAL PRONOUNS','AVG WORD LENGTH']))
df


# In[68]:


#df = pd.DataFrame({'Average sentence length':[result1],'Percentage of complex words':[result2],'Fog index':[result3], 'Average number of words per sentence':[result4],'Complex word count':[result5],'Word count':[result6],'Personal pronouns count':[result7],'Average word length':[result8]})
#df.info()
df

