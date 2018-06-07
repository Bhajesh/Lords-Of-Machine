import pandas as pd
import numpy as np
pd.options.display.max_rows=200
#pd.options.display.max_colwidth=400
from tqdm import tqdm, tqdm_pandas
import pdb
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import nltk
import re
import spacy
import string
import time
from nltk.corpus import stopwords

campaign_data = pd.read_csv('D:/WorkArea/Analytics Vidhya/Lords of Machine/train_HFxi8kT/campaign_data.csv')
train = pd.read_csv('D:/WorkArea/Analytics Vidhya/Lords of Machine/train_HFxi8kT/train.csv')
test = pd.read_csv('D:/WorkArea/Analytics Vidhya/Lords of Machine/test_BDIfz5B.csv/test_BDIfz5B.csv')


stop_words = set(stopwords.words('english'))
stop_words.union('-PRON-')
stop_words.union((string.punctuation))
nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])

def parseText(comment_text):
    com_text = ''.join(comment_text)
    try:
        lemmatize = []
        clean_text = re.sub('[^A-Za-z0-9]+', ' ',com_text.replace('\n',' ')).rstrip().lstrip().lower()
        f_text =  nlp(clean_text)
        for word in f_text:
            lemma = word.lemma_.strip()
            if lemma not in stop_words:
                if lemma not in lemmatize:
                    lemmatize.append(lemma)
        return ' '.join(lemmatize)
    except:
        return "Error occured, Pass text input to the function"
    

def word_to_code(df, column_nm):
    i = 100
    sub = dict()
    for idx, row in df.iterrows():
        row_split = row[column_nm].split()
        for word in row_split:
            if word not in sub.keys():
                i = i + 1
                sub[word] = i
    return sub

def createData(data):
    fl = []
    v = 0
    for idx, row in data.iterrows():
        list_data = []
        list_data.append([row.is_click, row.user_id, row.campaign_id,row.send_date_sec])
        sub_list = []
        row_split = row.formated_subject.split()

        l = []
        fill = len(row_split)
        while fill < 14:
            l.extend([v])
            fill = fill + 1
        
        for word in row_split:
            sub_list.extend([sub[word]])
      
        re_train_list = [item for sublist in list_data for item in sublist]
        tot = (re_train_list + sub_list + l)
        #print(idx)
        fl.append(tot)
    arr = np.array(fl)
    return arr

def createTestData(data):
    fl = []
    v = 0
    for idx, row in data.iterrows():
        list_data = []
        list_data.append([row.id, row.user_id, row.campaign_id,row.send_date_sec])
        sub_list = []
        row_split = row.formated_subject.split()

        l = []
        fill = len(row_split)
        while fill < 14:
            l.extend([v])
            fill = fill + 1
        
        for word in row_split:
            sub_list.extend([sub[word]])
      
        re_train_list = [item for sublist in list_data for item in sublist]
        tot = (re_train_list + sub_list + l)
        #print(idx)
        fl.append(tot)
    arr = np.array(fl)
    return arr

def to_predict(model, test_arr):
    p_list = []
    for _, row in enumerate(test_arr):
        v = np.array(row[1:]).reshape(1,17)
        v = v.astype(np.float)
        p = model.predict(v)
        p_list.append([row[0],int(p)])
    return p_list

def to_seconds(date):
    to_dt = pd.to_datetime(date)
    return time.mktime(to_dt.timetuple())
	
#, scoring='roc_auc'
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())