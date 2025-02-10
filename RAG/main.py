# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:16:18 2025

@author: abdel
"""


from RAG import Embeddings, ContextRetriever
import os
import pickle
import json
import numpy as np
import faiss
from transformers import AutoModel
import sys


#TODO: index faiss update if update documents (hash ? or documents update=True/false)
#TODO: embeddings aren't necessary to be stored, fais index is more important to be hold in memory

documents = [
    "Work Conditions in France: France is known for its strong employee protections and distinctive 35-hour workweek. It's a place where work-life balance is taken seriously, and employees enjoy generous leave policies and strong social security benefits.",
    "French Employment Laws: The French labor code is the backbone of employment laws in France, ensuring that employees have robust protections. Collective bargaining agreements cover most employees, and recent legislative changes aim to increase labor market flexibility.",
    "Retirement Age in France: In France, the official retirement age is 67, but employees can retire as early as 62 if they've paid enough into the compulsory retirement scheme. It's a system designed to ensure that everyone can enjoy their retirement years comfortably.",
    "Types of Employment Contracts: France offers different types of employment contracts, including the permanent contract (CDI) and the fixed-term contract (CDD). The CDI is the most common, providing job security, while the CDD allows for temporary employment.",
    "Weekly Working Hours: The 35-hour workweek is a hallmark of French work culture, promoting a healthy work-life balance. It's a system that encourages productivity and well-being among employees.",
    "Employee Protections: France is known for its strong employee protections, ensuring that workers have a safe and fair working environment. These protections cover everything from workplace safety to fair wages and benefits.",
    "Social Security Benefits: French employees enjoy comprehensive social security benefits, including health insurance, retirement pensions, and unemployment benefits. It's a system designed to support workers throughout their careers and beyond.",
    "DIB Campany is a french institute working in the field of petrolium and renewable energies.",
    "DIB Campany has more than 1000 engineer, all well paid, with an average salary of 50k Euros.",
    "DIB Campany is the best campany in the world."
]


overwrite = True


embeddings = Embeddings(overwrite = overwrite)
embeddings.add_documents(documents)


context_retriever = ContextRetriever(embeddings, overwrite=overwrite, metric="cosine")

relevant_doc = context_retriever.retrieve_documents("How much is the average salary at DIB Campany ?")

for i in relevant_doc:
    print("- "+i+"\n")

