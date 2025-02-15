# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:26:05 2025

@author: abdel
"""

import os
from PdfToText.PdfToText import PdfToText
from PdfToText.NougatParser import NougatParser
from RAG.RAG import Embeddings, ContextRetriever, ContextManager
from pathlib import Path
import json
from tqdm import tqdm
import time

# Define the paths
pdf_folder = Path(os.getcwd()+"/pdfDocuments")
cache_dir = Path(os.getcwd()+"/cache")
output_path_jsonl = cache_dir / f"pdfToTextCache/documents.jsonl"

# Get all PDF files in the folder
pdf_files = [pdf_folder/f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Convert PDFs to text and parse them
pdfToText = PdfToText()
parser = NougatParser(output_path_jsonl)

for file in pdf_files: #tqdm(pdf_files, desc="Converting pdf to text "):
    if not parser.is_in_parsed_files(file):
        generated_text = pdfToText.convert_by_batch(file, batch_size=2)
        parser.parse_document(markdown_text = generated_text, 
                              input_path    = str(file))
del pdfToText
time.sleep(1)


#Get all contents in one list
contents = parser.sections_to_lists(files_path=output_path_jsonl)
del parser
time.sleep(1)


# Create an Embeddings object to store the parsed documents
cache_dir_embeddings = cache_dir / "embeddings"
context_manager = ContextManager(documents = contents, overwrite = True, metric = "cosine",
                                  cache_dir = cache_dir_embeddings, embeddings_batch_size=2)


# Use the context to ask questions to the LLM
def ask_question(question):
    context = context_manager.retrieve_documents(question)
    
    response=context
    return response

# Test the ask_question function
answer = ask_question("Do you know the eqasim pipeline that is used to create MATSim scenarios? \
                      if yes, what is the discrete mode choice and what are the utility functions that define it ?")

print("\n")
for i in answer:
    print("- "+i)