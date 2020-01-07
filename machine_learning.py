# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 16:22:11 2019

@author: AITICHOU
"""


##########################################################################################
#Importation des modules et définition des fonctions
##########################################################################################

import pandas as pd
import numpy as np
import random
from collections import Counter
from nltk.corpus import stopwords
import string

from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


import spacy
nlp = spacy.load('fr')

##########################

def read_df(file):
    df = pd.read_csv(file, sep=",",decimal=".", encoding="utf-8")
    return df

       
def cleaner(text):
    t = [i for i in word_tokenize(text.lower()) if i not in stop]
    recette_clean = " ".join(t)
    return recette_clean
    
def lemmatise_text(text):
    text = nlp(text)
    lemmas = [token.lemma_ for token in text]
    return ' '.join(lemmas)
    
def stem_text(text):
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    stemmer = SnowballStemmer('french')
    stems = [stemmer.stem(token) for token in tokenizer.tokenize(text)]
    return ' '.join(stems)
    
def print_n_strongly_associated_features(vectoriser, model, n):
    feature_names = np.array(vectoriser.get_feature_names())

    for i in range(3):
        class_name = model.classes_[i]
        print("CLASSE {}".format(class_name))
        idx_coefs_sorted = model.coef_[i].argsort()
        print("Les dix variables ayant l'association négative la plus forte " + 
              "avec la classe {} :\n{}\n".format(class_name,
                                                 feature_names[idx_coefs_sorted[:n]]))
        print("Les dix variables ayant l'association positive la plus forte " +
              "avec la classe {} :\n{}\n"
              .format(class_name,
                      feature_names[idx_coefs_sorted[:-(n + 1):-1]]))
        print()

def run_forests(X_train,Y_train,X_test,Y_test):    
    print('random forest: \n')   
    params = []
    scores = []
    
    liste_estimators = [50,100, 500, 1000]
    for estim in liste_estimators:
        forest = RandomForestClassifier(n_estimators=estim,
                                        max_features=None,
                                        max_depth=None)                                   
        forest_fit = forest.fit(X_train, Y_train)
        pred = forest_fit.predict(X_test)
        print('\n params:', estim)
        print('forest train: ',accuracy_score(Y_train, forest_fit.predict(X_train)), ' test: ',
              accuracy_score(Y_test, pred))
        params.append( estim )
        scores.append( accuracy_score(Y_test, pred))

    print('best:', params[np.argmax(scores)]) 


##########################################################################################
#Exploration des données
##########################################################################################

class_distribution = (pd.DataFrame.from_dict(Counter(df.categorie.values), orient='index').rename(columns={0: 'effectif'}))
class_distribution.index.name = 'categorie'
class_distribution['pourcentage'] = np.around(class_distribution.effectif / np.sum(class_distribution.effectif), 2)
class_distribution

#données équilibrées pas de rééquilibrage à faire


##########################################################################################
#Nettoyage des données
##########################################################################################

stop_words = set(stopwords.words("french"))
stop_words = list(stop_words)

for l in list(string.ascii_letters):
    stop_words.append(l)


#On ajoute des "mots vides" à la liste de stop_words
stop_words.append('les')
stop_words.append('être')
stop_words.append('avoir')
stop_words.append('faire')
print(stop_words)

stop = stop_words + list(string.punctuation)
print(stop)

#Suppression des mots vides et ponctuation
df['recette_clean'] = df['preparation'].apply(cleaner)
df.head(10)

#Lemmatisation
df['lemmas'] = df['recette_clean'].apply(lemmatise_text)
df.head(10)

#Racinisation
df['stems'] = df['recette_clean'].apply(stem_text)
df.head(10)

#Séparation du jeu de données
X_train, X_valid, y_train, y_valid = train_test_split(df['recette_clean'], df['categorie'], train_size=0.75, random_state=1234)
X_train.shape, X_valid.shape

X_train_stems, X_valid_stems, y_train_stems, y_valid_stems = train_test_split(df['stems'], df['categorie'], train_size=0.75, random_state=1234)
X_train_stems.shape, X_valid_stems.shape

X_train_lem, X_valid_lem, y_train_lem, y_valid_lem = train_test_split(df['lemmas'], df['categorie'], train_size=0.75, random_state=1234)
X_train_lem.shape, X_valid_lem.shape


##########################################################################################
#Préparation des données
##########################################################################################

#Extraction des descripeurs

# Calcul des fréquences d'occurrence des termes dans le corpus, avec les options par défaut
vect_count = CountVectorizer().fit(X_train)
vect_count

vect_count_stems = CountVectorizer().fit(X_train_stems)

vect_count_lem = CountVectorizer().fit(X_train_lem)

#Examinons le vocabulaire de notre corpus
vect_count.get_feature_names()[:10]
vect_count_lem.get_feature_names()[-10:]

#taille du vocabulaire
len(vect_count.get_feature_names())
len(vect_count_stems.get_feature_names())
len(vect_count_lem.get_feature_names())

#Création de la matrice document-termes
X_train_vectorized_count = vect_count.transform(X_train)
X_train_vectorized_count

X_train_stems_vectorized_count = vect_count_stems.transform(X_train_stems)
X_train_stems_vectorized_count

X_train_lem_vectorized_count = vect_count_lem.transform(X_train_lem)
X_train_lem_vectorized_count

# Le corpus de validation doit être également transformé en matrice document-termes.
#Les termes sont ceux décomptés sur le corpus d'entraînement. Les termes présent dans le corpus
#de validation mais absents du corpus d'entraînement seront ignorés.
X_valid_vectorized_count = vect_count.transform(X_valid)

X_valid_stems_vectorized_count = vect_count_stems.transform(X_valid_stems)

X_valid_lem_vectorized_count = vect_count_lem.transform(X_valid_lem)


#Cette fois-ci nous allons inclure des bigrammes dans le vocabulaire:
vect_count_bigrams = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)
X_train_vectorized_count_bigrams = vect_count_bigrams.transform(X_train)
X_valid_vectorized_count_bigrams = vect_count_bigrams.transform(X_valid)
len(vect_count_bigrams.get_feature_names())

vect_count_bigrams_stems = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train_stems)
X_train_vectorized_count_bigrams_stems = vect_count_bigrams_stems.transform(X_train_stems)
X_valid_vectorized_count_bigrams_stems = vect_count_bigrams_stems.transform(X_valid_stems)
len(vect_count_bigrams_stems.get_feature_names())

vect_count_bigrams_lem = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train_lem)
X_train_vectorized_count_bigrams_lem = vect_count_bigrams_lem.transform(X_train_lem)
X_valid_vectorized_count_bigrams_lem = vect_count_bigrams_lem.transform(X_valid_lem)
len(vect_count_bigrams_lem.get_feature_names())


#On inclut les trigrammes dans le vocabulaire
vect_count_trigrams = CountVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
X_train_vectorized_count_trigrams = vect_count_trigrams.transform(X_train)
X_valid_vectorized_count_trigrams = vect_count_trigrams.transform(X_valid)
len(vect_count_trigrams.get_feature_names())

vect_count_trigrams_stems = CountVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train_stems)
X_train_vectorized_count_trigrams_stems = vect_count_trigrams_stems.transform(X_train_stems)
X_valid_vectorized_count_trigrams_stems = vect_count_trigrams_stems.transform(X_valid_stems)
len(vect_count_trigrams_stems.get_feature_names())

vect_count_trigrams_lem = CountVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train_lem)
X_train_vectorized_count_trigrams_lem = vect_count_trigrams_lem.transform(X_train_lem)
X_valid_vectorized_count_trigrams_lem = vect_count_trigrams_lem.transform(X_valid_lem)
len(vect_count_trigrams_lem.get_feature_names())

#On inclut les quadrigrammes dans le vocabulaire
vect_count_quadrigrams = CountVectorizer(min_df=5, ngram_range=(1,4)).fit(X_train)
X_train_vectorized_count_quadrigrams = vect_count_quadrigrams.transform(X_train)
X_valid_vectorized_count_quadrigrams = vect_count_quadrigrams.transform(X_valid)
len(vect_count_quadrigrams.get_feature_names())

vect_count_quadrigrams_stems = CountVectorizer(min_df=5, ngram_range=(1,4)).fit(X_train_stems)
X_train_vectorized_count_quadrigrams_stems = vect_count_quadrigrams_stems.transform(X_train_stems)
X_valid_vectorized_count_quadrigrams_stems = vect_count_quadrigrams_stems.transform(X_valid_stems)
len(vect_count_quadrigrams_stems.get_feature_names())

vect_count_quadrigrams_lem = CountVectorizer(min_df=5, ngram_range=(1,4)).fit(X_train_lem)
X_train_vectorized_count_quadrigrams_lem = vect_count_quadrigrams_lem.transform(X_train_lem)
X_valid_vectorized_count_quadrigrams_lem = vect_count_quadrigrams_lem.transform(X_valid_lem)
len(vect_count_quadrigrams_lem.get_feature_names())

#On inclut les quintigrammes dans le vocabulaire
vect_count_quintigrams = CountVectorizer(min_df=5, ngram_range=(1,5)).fit(X_train)
X_train_vectorized_count_quintigrams = vect_count_quintigrams.transform(X_train)
X_valid_vectorized_count_quintigrams = vect_count_quintigrams.transform(X_valid)
len(vect_count_quintigrams.get_feature_names())

vect_count_quintigrams_stems = CountVectorizer(min_df=5, ngram_range=(1,5)).fit(X_train_stems)
X_train_vectorized_count_quintigrams_stems = vect_count_quintigrams_stems.transform(X_train_stems)
X_valid_vectorized_count_quintigrams_stems = vect_count_quintigrams_stems.transform(X_valid_stems)
len(vect_count_quintigrams_stems.get_feature_names())

vect_count_quintigrams_lem = CountVectorizer(min_df=5, ngram_range=(1,5)).fit(X_train_lem)
X_train_vectorized_count_quintigrams_lem = vect_count_quintigrams_lem.transform(X_train_lem)
X_valid_vectorized_count_quintigrams_lem = vect_count_quintigrams_lem.transform(X_valid_lem)
len(vect_count_quintigrams_lem.get_feature_names())


#Numérique continu : TF-IDF

#Limitons le vocabulaire à des termes qui apparaissent dans au moins 5 recettes
vect_tfidf = TfidfVectorizer(min_df=5).fit(X_train)

len(vect_count.get_feature_names()), len(vect_tfidf.get_feature_names())

X_train_vectorized_tfidf = vect_tfidf.transform(X_train)
X_valid_vectorized_tfidf = vect_tfidf.transform(X_valid)
X_train_vectorized_tfidf_stems = vect_tfidf.transform(X_train_stems)
X_valid_vectorized_tfidf_stems = vect_tfidf.transform(X_valid_stems)
X_train_vectorized_tfidf_lem = vect_tfidf.transform(X_train_lem)
X_valid_vectorized_tfidf_lem = vect_tfidf.transform(X_valid_lem)

tfidf_transformer = TfidfTransformer()

X_train_tfidf_bigrams = tfidf_transformer.fit_transform(X_train_vectorized_count_bigrams)
X_valid_tfidf_bigrams = tfidf_transformer.fit_transform(X_valid_vectorized_count_bigrams)
X_train_tfidf_bigrams_stems = tfidf_transformer.fit_transform(X_train_vectorized_count_bigrams_stems)
X_valid_tfidf_bigrams_stems = tfidf_transformer.fit_transform(X_valid_vectorized_count_bigrams_stems)
X_train_tfidf_bigrams_lem = tfidf_transformer.fit_transform(X_train_vectorized_count_bigrams_lem)
X_valid_tfidf_bigrams_lem = tfidf_transformer.fit_transform(X_valid_vectorized_count_bigrams_lem)

X_train_tfidf_trigrams = tfidf_transformer.fit_transform(X_train_vectorized_count_trigrams)
X_valid_tfidf_trigrams = tfidf_transformer.fit_transform(X_valid_vectorized_count_trigrams)
X_train_tfidf_trigrams_stems = tfidf_transformer.fit_transform(X_train_vectorized_count_trigrams_stems)
X_valid_tfidf_trigrams_stems = tfidf_transformer.fit_transform(X_valid_vectorized_count_trigrams_stems)
X_train_tfidf_trigrams_lem = tfidf_transformer.fit_transform(X_train_vectorized_count_trigrams_lem)
X_valid_tfidf_trigrams_lem = tfidf_transformer.fit_transform(X_valid_vectorized_count_trigrams_lem)

X_train_tfidf_quadrigrams = tfidf_transformer.fit_transform(X_train_vectorized_count_quadrigrams)
X_valid_tfidf_quadrigrams = tfidf_transformer.fit_transform(X_valid_vectorized_count_quadrigrams)
X_train_tfidf_quadrigrams_stems = tfidf_transformer.fit_transform(X_train_vectorized_count_quadrigrams_stems)
X_valid_tfidf_quadrigrams_stems = tfidf_transformer.fit_transform(X_valid_vectorized_count_quadrigrams_stems)
X_train_tfidf_quadrigrams_lem = tfidf_transformer.fit_transform(X_train_vectorized_count_quadrigrams_lem)
X_valid_tfidf_quadrigrams_lem = tfidf_transformer.fit_transform(X_valid_vectorized_count_quadrigrams_lem)

X_train_tfidf_quintigrams = tfidf_transformer.fit_transform(X_train_vectorized_count_quintigrams)
X_valid_tfidf_quintigrams = tfidf_transformer.fit_transform(X_valid_vectorized_count_quintigrams)
X_train_tfidf_quintigrams_stems = tfidf_transformer.fit_transform(X_train_vectorized_count_quintigrams_stems)
X_valid_tfidf_quintigrams_stems = tfidf_transformer.fit_transform(X_valid_vectorized_count_quintigrams_stems)
X_train_tfidf_quintigrams_lem = tfidf_transformer.fit_transform(X_train_vectorized_count_quintigrams_lem)
X_valid_tfidf_quintigrams_lem = tfidf_transformer.fit_transform(X_valid_vectorized_count_quintigrams_lem)



##########################################################################################
#Machine Learning
##########################################################################################

#Classifieur naïf bayesien (En général pris comme baseline.)

model_nb = MultinomialNB().fit(X_train_vectorized_tfidf, y_train)
predictions_valid = model_nb.predict(X_valid_vectorized_tfidf)
accuracy_score(y_valid, predictions_valid)

print(classification_report(y_valid, predictions_valid))


####################
# Regression logistique
####################


#Unigrammes

model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_vectorized_tfidf, y_train)
predictions_valid = model_lr.predict(X_valid_vectorized_tfidf)
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))



   #Données Racinisées
model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_vectorized_tfidf_stems, y_train_stems)
predictions_valid = model_lr.predict(X_valid_vectorized_tfidf_stems)
accuracy_score(y_valid_stems, predictions_valid)
print(classification_report(y_valid_stems, predictions_valid))


   #Données Lemmatisées
model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_vectorized_tfidf_lem, y_train_lem)
predictions_valid = model_lr.predict(X_valid_vectorized_tfidf_lem)
accuracy_score(y_valid_lem, predictions_valid)
print(classification_report(y_valid_lem, predictions_valid))


#Avec le vectorisateur à unigrammes et bigrammes :

model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_bigrams,
                                                  y_train)
predictions_valid = model_lr.predict(X_valid_tfidf_bigrams )
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))
print_n_strongly_associated_features(vect_count_bigrams, model_lr, 10)


    #Données Racinisées
model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_bigrams_stems,
                                                  y_train_stems)
predictions_valid = model_lr.predict(X_valid_tfidf_bigrams_stems )
accuracy_score(y_valid_stems, predictions_valid)
print(classification_report(y_valid_stems, predictions_valid))


    #Données Lemmatisées
model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_bigrams_lem,
                                                  y_train_lem)
predictions_valid = model_lr.predict(X_valid_tfidf_bigrams_lem )
accuracy_score(y_valid_lem, predictions_valid)
print(classification_report(y_valid_lem, predictions_valid))



#Avec le vectorisateur à unigrammes, bigrammes et trigrammes:

model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_trigrams,
                                                  y_train)
predictions_valid = model_lr.predict(X_valid_tfidf_trigrams )
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))


    #Données Racinisées
model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_trigrams_stems,
                                                  y_train_stems)
predictions_valid = model_lr.predict(X_valid_tfidf_trigrams_stems )
accuracy_score(y_valid_stems, predictions_valid)
print(classification_report(y_valid_stems, predictions_valid))


    #Données Lemmatisées
model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_trigrams_lem,
                                                  y_train_lem)
predictions_valid = model_lr.predict(X_valid_tfidf_trigrams_lem )
accuracy_score(y_valid_lem, predictions_valid)
print(classification_report(y_valid_lem, predictions_valid))


#Avec le vectorisateur à unigrammes, bigrammes, trigrammes et quadrigrammes:

model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_quadrigrams,
                                                  y_train)
predictions_valid = model_lr.predict(X_valid_tfidf_quadrigrams )
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))


    #Données Racinisées
model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_quadrigrams_stems,
                                                  y_train_stems)
predictions_valid = model_lr.predict(X_valid_tfidf_quadrigrams_stems )
accuracy_score(y_valid_stems, predictions_valid)
print(classification_report(y_valid_stems, predictions_valid))


    #Données Lemmatisées
model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_quadrigrams_lem,
                                                  y_train_lem)
predictions_valid = model_lr.predict(X_valid_tfidf_quadrigrams_lem )
accuracy_score(y_valid_lem, predictions_valid)
print(classification_report(y_valid_lem, predictions_valid))


#Avec le vectorisateur à unigrammes, bigrammes, trigrammes, quadrigrammes et quintigrammes:

model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_quintigrams,
                                                  y_train)
predictions_valid = model_lr.predict(X_valid_tfidf_quintigrams )
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))


    #Données Racinisées
model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_quintigrams_stems,
                                                  y_train_stems)
predictions_valid = model_lr.predict(X_valid_tfidf_quintigrams_stems )
accuracy_score(y_valid_stems, predictions_valid)
print(classification_report(y_valid_stems, predictions_valid))


    #Données Lemmatisées
model_lr = LogisticRegression(multi_class='multinomial',
                              solver='lbfgs').fit(X_train_tfidf_quintigrams_lem,
                                                  y_train_lem)
predictions_valid = model_lr.predict(X_valid_tfidf_quintigrams_lem )
accuracy_score(y_valid_lem, predictions_valid)
print(classification_report(y_valid_lem, predictions_valid))

###############
###SVM
###############

#noyau linéaire

model_svm = SVC(kernel='linear', C=0.1).fit(X_train_vectorized_tfidf, y_train)
predictions_valid = model_svm.predict(X_valid_vectorized_tfidf)
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))

model_svm = SVC(kernel='linear', C=0.1).fit(X_train_tfidf_bigrams, y_train)
predictions_valid = model_svm.predict(X_valid_tfidf_bigrams)
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))

model_svm = SVC(kernel='linear', C=0.1).fit(X_train_tfidf_trigrams, y_train)
predictions_valid = model_svm.predict(X_valid_tfidf_trigrams)
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))

model_svm = SVC(kernel='linear', C=0.1).fit(X_train_tfidf_quadrigrams, y_train)
predictions_valid = model_svm.predict(X_valid_tfidf_quadrigrams)
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))

model_svm = SVC(kernel='linear', C=0.1).fit(X_train_tfidf_quintigrams, y_train)
predictions_valid = model_svm.predict(X_valid_tfidf_quintigrams)
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))


#noyau radial
model_svm = SVC(kernel='rbf', C=0.1).fit(X_train_tfidf_bigrams, y_train)
predictions_valid = model_svm.predict(X_valid_tfidf_bigrams)
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))


#noyau polynomial (degré 3 par défaut)
model_svm = SVC(kernel='poly', C=0.1).fit(X_train_tfidf_bigrams, y_train)
predictions_valid = model_svm.predict(X_valid_tfidf_bigrams)
accuracy_score(y_valid, predictions_valid)
print(classification_report(y_valid, predictions_valid))


################
###Random Forest
################

model_rf = run_forests(X_train_vectorized_tfidf,y_train,X_valid_vectorized_tfidf,y_valid)

model_rf = run_forests(X_train_vectorized_count_bigrams,y_train,X_valid_vectorized_count_bigrams,y_valid)

model_rf = run_forests(X_train_vectorized_count_trigrams,y_train,X_valid_vectorized_count_trigrams,y_valid)
