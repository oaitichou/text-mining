# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 12:36:06 2019

@author: AITICHOU
"""

import pandas as pd

import string
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

tokenizer = PunktSentenceTokenizer('tokenizers/punkt/PY3/french.pickle')


import spacy
nlp = spacy.load('fr')

def read_df(file):
    df = pd.read_csv(file, sep=",",decimal=".", encoding="utf-8")
    return df
    

    
def replace_words_with_pos_tag(text):
    text = nlp(text)
    return ' '.join([token.pos_ for token in text])


df_dessert = pd.read_csv('C:/Users/AITICHOU/Desktop/marmiton/base_dessert_marmiton.csv', sep=";",decimal=".", encoding='utf-8')
df_dessert = df_dessert.dropna()
df_dessert.head()

df_entree = pd.read_csv('C:/Users/AITICHOU/Desktop/marmiton/base_entree_marmiton.csv', sep=";",decimal=".", encoding='utf-8')
df_entree = df_entree.dropna()
df_entree.head()

df_plat = pd.read_csv('C:/Users/AITICHOU/Desktop/marmiton/base_plat_marmiton.csv', sep=";",decimal=".", encoding='utf-8')
df_plat = df_plat.dropna()
df_plat.head()





prep_entree = df_entree.preparation
prep_plat = df_plat.preparation
prep_dessert = df_dessert.preparation



#création de corpus 

text_entree = ' '.join(a for a in prep_entree)
text_plat = ' '.join(a for a in prep_plat)
text_dessert = ' '.join(a for a in prep_dessert)



#nettoyage des données

phrases_entree=tokenizer.tokenize(text_entree)
print(phrases_entree[:10])


phrases_plat=tokenizer.tokenize(text_plat)
phrases_dessert=tokenizer.tokenize(text_dessert)


#ETAPE 2 :
# Mots et mots vides

tokenizer2=RegexpTokenizer('[\s+\'\.\,\?\!();\:\"\[\-\]\\\\/]',gaps=True)
stop_words=set(stopwords.words('french'))
stop_words = list(stop_words)

for l in list(string.ascii_letters):
    stop_words.append(l)


stop_words.append('les')
stop_words.append('être')
stop_words.append('avoir')
stop_words.append('faire')
print(stop_words)


def ngramme_bon(n,liste_phrases):
    dic_trigramme={}
    for p in liste_phrases:
        mots=tokenizer2.tokenize(str(p))
        for i in range(0,len(mots)-(n-1)):
            trigramme = mots[i].lower()
            gramme_total = [mots[i].lower()]
            for j in range(1,n):
                trigramme+=' '+mots[i+j].lower()
                gramme_total.append(mots[i+j].lower())
            
            gramme = []
            for elem in gramme_total :
                if elem.lower() not in stop_words:
                    gramme.append(elem)
            if len(gramme)/len(gramme_total)>0.5 :
                if trigramme in dic_trigramme.keys():
                    dic_trigramme[trigramme]+=1
                else :
                    dic_trigramme[trigramme]=1
    return(dic_trigramme)

bigramme_entree = ngramme_bon(2,phrases_entree)
bigramme_entree_tri = sorted(bigramme_entree.items(),key=lambda x:x[1],reverse=True)

bigramme_plat = ngramme_bon(2,phrases_plat)
bigramme_plat_tri = sorted(bigramme_plat.items(),key=lambda x:x[1],reverse=True)


bigramme_dessert = ngramme_bon(2,phrases_dessert)
bigramme_dessert_tri = sorted(bigramme_dessert.items(),key=lambda x:x[1],reverse=True)

#Ecriture dans les fichiers


NewFichier= open('C:/Users/AITICHOU/Desktop/marmiton/bigramme_entree.txt','w',encoding='utf-8')
for tupl in bigramme_entree_tri:
    NewFichier.write("{};{}\n".format(tupl[0],tupl[1]))
    
NewFichier.close()

NewFichier= open('C:/Users/AITICHOU/Desktop/marmiton/bigramme_plat.txt','w',encoding='utf-8')
for tupl in bigramme_plat_tri:
    NewFichier.write("{};{}\n".format(tupl[0],tupl[1]))
    
NewFichier.close()

NewFichier= open('C:/Users/AITICHOU/Desktop/marmiton/bigramme_dessert.txt','w',encoding='utf-8')
for tupl in bigramme_dessert_tri:
    NewFichier.write("{};{}\n".format(tupl[0],tupl[1]))
    
NewFichier.close()


df = pd.concat([df_entree,df_plat,df_dessert])
df_box = df.drop(['nom_recette','ingredients','preparation'], axis=1)
df_box = df_box.set_index('categorie') 
df_box.T.boxplot()

from statistics import mean, median

somelist = df_dessert.nb_etape

max_value = max(somelist)
min_value = min(somelist)
avg_value = sum(somelist)/len(somelist)
median_value = median(somelist)