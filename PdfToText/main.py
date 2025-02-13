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
    
# Initialize the PdfToText and parser classes
pdfToText = PdfToText()
parser    = NougatParser()

# Convert the PDF to text
generated_text = pdfToText.convert_by_batch(file_path, batch_size=8)
parser.parse_document(generated_text, output_path_jsonl)








