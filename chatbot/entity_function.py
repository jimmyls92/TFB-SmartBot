# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 18:36:48 2021

@author: luism, Jaime
"""

import spacy
def getEntities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities_list = [(X.text) for X in doc.ents]
    print(entities_list)
    if len(entities_list)>1:
        # devuelve 1 si hay más de una entidad
        entity = 1
    else:
        # devuelve 0 si no ninguna entidad
        try:
            entity = entities_list[0]
        except IndexError:
            entity = 0
    return entity