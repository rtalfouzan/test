from GoogleNews import GoogleNews
import pandas as pd
import numpy as np
import spacy
import string
import re
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request
import json

nlp = spacy.load("spacy.aravec.model")

app = Flask(__name__)

def clean_str(text):
    try:
        search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
        replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']
        
        #remove tashkeel
        p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        text = re.sub(p_tashkeel,"", text)
        
        #remove longation
        p_longation = re.compile(r'(.)\1+')
        subst = r"\1\1"
        text = re.sub(p_longation, subst, text)
        
        text = text.replace('وو', 'و')
        text = text.replace('يي', 'ي')
        text = text.replace('اا', 'ا')
        
        for i in range(0, len(search)):
            text = text.replace(search[i], replace[i])
        
        #trim    
        text = text.strip()

        return text
    
    except Exception as e:
        return {"error":str(e)}


def split_hashtag_to_words(tag):
    try:
        tag = tag.replace('#','')
        tags = tag.split('_')
        if len(tags) > 1 :
            
            return tags
        pattern = re.compile(r"[A-Z][a-z]+|\d+|[A-Z]+(?![a-z])")
        return pattern.findall(tag)
    except Exception as e:
        return {"error": str(e)}

def clean_hashtag(text):
    try:
        words = text.split()
        text = list()
        for word in words:
            if is_hashtag(word):
                text.extend(extract_hashtag(word))
            else:
                text.append(word)
        return " ".join(text)
    except Exception as e:
        return {"error":str(e)}

def is_hashtag(word):
    try:
        if word.startswith("#"):
            return True
        else:
            return False
    except Exception as e:
        return {"error":str(e)}

def extract_hashtag(text):
    try:
        hash_list = ([re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")])
        word_list = []
        for word in hash_list :
            word_list.extend(split_hashtag_to_words(word))
        return word_list
    except Exception as e:
        return {"error":str(e)}

# Define the preprocessing Class
class Preprocessor:
    def __init__(self, tokenizer, **cfg):
        self.tokenizer = tokenizer

    def __call__(self, text):
        preprocessed = clean_str(text)
        return self.tokenizer(preprocessed)   
      
# Apply the `Preprocessor` Class
nlp.tokenizer = Preprocessor(nlp.tokenizer)
      
#--------------------------------------------------------------------------------------------------------------- 
#----------------------------------------- END OF PRE-PROCESSING------------------------------------------------ 
#--------------------------------------------------------------------------------------------------------------- 
@app.route('/fakenews/',methods=['POST'])
def fakenews():
    param = json.loads(request.data)
    if 'input' in param:

        tx = param['input']

        googlenews = GoogleNews(lang='ar')
        googlenews.clear()

        f =0 
        Prediction =''
        top_similar_ind =''
        top_similar_news =''
        medium =''
        top_similar_ind2 =''
        tp_desc =''

        tx = clean_hashtag(tx)
        tx = clean_str(tx) 


        googlenews.search(tx)
        result = googlenews.page_at(1)

        if len(result) == 0:
            Prediction ='Fake'
            top_similar_news ='No sim news'
            medium ='No Source'
            tp_desc ='No desc available'

        else:
            result_text = {"Text":[]}
                #google search
            for i in range(len(result)):
                title =result[i]['title']
                result_text['Text'].append(title)

            result_text2 = {"Text":[]}
                #google search
            for i in range(len(result)):
                desc =result[i]['desc']
                result_text2['Text'].append(desc) 
            
        googlenews.clear()

        result_text = pd.DataFrame(result_text)
        result_text2 = pd.DataFrame(result_text2)

        data = pd.DataFrame()

        data['Text2'] = result_text['Text'].copy()
        data['Text2'] = data['Text2'].apply(lambda x: nlp(x).similarity(nlp(tx)))

        sg300top = data['Text2'].max(axis = 0)

        top_similar_ind = np.argmax(data['Text2'])
        top_similar_news = result[top_similar_ind]['title']
        medium = result[top_similar_ind]['media']


        data['Text3'] = result_text2['Text'].copy()
        data['Text3'] = data['Text3'].apply(lambda x: nlp(x).similarity(nlp(tx)))

        sg300top2 = data['Text3'].max(axis = 0)

        top_similar_ind2 = np.argmax(data['Text3'])
        tp_desc = result[top_similar_ind2]['desc']

        if sg300top >= .85 or sg300top2 >= .85 :
            Prediction ='True'
        else:
            Prediction ='False'

        output = {"prediction":Prediction,"top_news_title":(top_similar_news),"medium":(medium),"top_news_description":(tp_desc)}

        return output
    
    else:
        return {"error":"wrong input"}

if __name__ == '__main__':
    app.run()