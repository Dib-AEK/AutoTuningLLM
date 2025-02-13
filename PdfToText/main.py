# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:42:17 2025

@author: abdel
"""

import os
from pdfToText import PdfToText
from pathlib import Path
from NougatParser import NougatParser

#TODO: Add overwrite option

file_path = Path("../pdfDocuments/Pairing discrete mode choice models and agent-based transport simulation with MATSim.pdf")
cache_dir = Path("../cache/pdfToTextCache")
output_path = cache_dir / f"{file_path.stem}.txt"
output_path_jsonl = cache_dir / f"{file_path.stem}.jsonl"

# if not os.path.exists(cache_dir): os.makedirs(cache_dir, exist_ok=True)
    
# Initialize the PdfToText class
pdfToText = PdfToText()

# Convert the PDF to text
generated_text = pdfToText.convert_by_batch(file_path, batch_size=8)

# Write the generated content to the new text file
try:
    with output_path.open("w", encoding="utf-8") as file:
        file.write("".join(generated_text))  # Concatenate all elements (pages) of `generated_text` into a single string
    print(f"Text successfully written to: {output_path}")
except Exception as e:
    print(f"An error occurred: {e}")


with open(output_path, 'r', encoding='utf-8') as file:
    generated_text = file.read() 

parser = NougatParser(generated_text)
parser.save_to_jsonl(output_path_jsonl)








