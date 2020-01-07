# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 22:35:33 2019

@author: AITICHOU
"""

import urllib.request as req
import re
import bs4 as BS
import numpy as np
import os
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#on va importer les avis des spectateurs des films sortis en 2017 sur Allociné

#Extraire le nombre de pages de la catégorie entree
url = 'https://www.marmiton.org/recettes/?type=dessert'
html = req.urlopen(url)
soup = BS.BeautifulSoup(html,"html.parser")

nb_pages = 30

#Extraire les url des pages de la catégorie 2017
liste_url_pages = []
for i in range(1, nb_pages+1):
    url0 = url+'&page='+str(i)
    liste_url_pages.append(url0)
print(len(liste_url_pages))


#Récupération des Urls de recettes
#trop long de récupérer toutes les recettes 

liste_url_recette = []
for url in liste_url_pages :
    html = req.urlopen(url)
    soup = BS.BeautifulSoup(html,"html.parser")
    liste_recette = soup.find_all('a',attrs={"class":u"recipe-card-link"})
    for recette in liste_recette :
        url_recette = recette['href']
        liste_url_recette.append(url_recette)


#récupération du titre de la recette et de la rectte en elle même

lst_categorie = []
lst_titre = []
lst_ingredients = []
lst_nb_etape = []
lst_prepa = []

############A compiler"

i = 0
for recette in liste_url_recette :
    html = req.urlopen(recette)
    soup = BS.BeautifulSoup(html,"html.parser")
    titre = soup.find('h1',attrs={"class":u"main-title"}).get_text()
    ingredients_items = soup.find_all('li',attrs={"class":u"recipe-ingredients__list__item"})
    ingredients = []
    for ingredient in ingredients_items :
        texte = ingredient.get_text()
        texte = re.sub("\n","",texte)
        ingredients.append(texte)

    preparation_items = soup.find_all('li',attrs={"class":u"recipe-preparation__list__item"})

    nb_etape = len(preparation_items)
    prepa = ''
    for etape in preparation_items :
        text_etape = etape.find_next('h3',attrs={"class":u"__secondary"}).get_text()
        texte = re.sub(text_etape+'(\s)+' ,'', etape.get_text())
        texte = re.sub('(\n|\t)','',texte)
        prepa+= ' '+texte
    
    lst_categorie.append('dessert')
    lst_titre.append(titre)
    lst_ingredients.append(ingredients)
    lst_nb_etape.append(nb_etape)
    lst_prepa.append(prepa)
    i+=1
    print(i)



df = pd.DataFrame()
df['categorie'] = lst_categorie
df['nom_recette'] = lst_titre
df['ingredients']= lst_ingredients
df['preparation']= lst_prepa
df['nb_etape'] = lst_nb_etape


import csv

df.to_csv('C:/Users/AITICHOU/Desktop/marmiton/base_dessert_marmiton.csv',encoding='utf-8',na_rep="NA",sep=';',index=False)
df.to_csv('C:/Users/AITICHOU/Desktop/marmiton/base_dessert_marmiton_v2.csv',encoding='utf-8',na_rep="NA",sep=';',quoting=csv.QUOTE_NONE, escapechar='\\',index=False)

