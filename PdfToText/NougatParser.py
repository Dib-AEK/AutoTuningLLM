# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:19:55 2025

@author: abdel
"""
import re
import json
from bs4 import BeautifulSoup
import markdown
from typing import Dict, List, Union

class NougatParser:
    def __init__(self):
        self.parsed_data = {}
    
    def parse_document(self, markdown_text: str, output_path: str = None):
        self.parsed_data = {
            "title": "",
            "authors": [],
            "abstract": "",
            "sections": []
        }
        
        html = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html, "html.parser")

        self._extract_title(soup)
        self._extract_authors(soup)
        self._extract_abstract(soup)
        self._extract_sections(soup)
        
        if output_path:
            self.save_to_jsonl(output_path)
    
    def _extract_title(self, soup):
        first_header = soup.find("h1")
        if first_header:
            self.parsed_data["title"] = first_header.text.strip()
            first_header.extract()
    
    def _extract_authors(self, soup):
        authors = []
        for paragraph in soup.find_all("p"):
            text = paragraph.text.strip()
            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', text):  # Simple heuristic for names
                authors.append(text)
            elif "@" in text:  # Assuming emails are part of author info
                continue
            else:
                break
        self.parsed_data["authors"] = authors
    
    def _extract_abstract(self, soup):
        abstract_header = soup.find("h6")
        if abstract_header and abstract_header.text.strip().lower() == "abstract":
            abstract_paragraph = abstract_header.find_next("p")
            if abstract_paragraph:
                self.parsed_data["abstract"] = abstract_paragraph.text.strip()
                abstract_paragraph.extract()
    
    def _extract_sections(self, soup):
        current_section = {"heading": "", "content": ""}
        
        for tag in soup.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol']):
            if tag.name in ['h2', 'h3', 'h4']:
                if current_section["heading"] and not self._contains_ignored_content(current_section["content"]):
                    self.parsed_data["sections"].append(current_section)
                current_section = {"heading": tag.text.strip(), "content": ""}
            elif tag.name == 'p':
                strong_tag = tag.find('strong')
                if strong_tag and re.match(r'^\d+\.\s+', strong_tag.text.strip()):
                    if current_section["heading"] and not self._contains_ignored_content(current_section["content"]):
                        self.parsed_data["sections"].append(current_section)
                    current_section = {"heading": strong_tag.text.strip(), "content": ""}
                    strong_tag.extract()
                else:
                    text = tag.text.strip()
                    if self._is_potential_section_heading(text):
                        if current_section["heading"] and not self._contains_ignored_content(current_section["content"]):
                            self.parsed_data["sections"].append(current_section)
                        current_section = {"heading": text, "content": ""}
                    else:
                        current_section["content"] += text + "\n"
            else:
                current_section["content"] += tag.text.strip() + "\n"
        
        if current_section["heading"] and not self._contains_ignored_content(current_section["content"]):
            self.parsed_data["sections"].append(current_section)
    
    def _is_potential_section_heading(self, text):
        return len(text.split()) <= 5 and text.istitle()
    
    def _contains_ignored_content(self, content):
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        return "@" in content or "orcid" in content.lower() or re.search(phone_pattern, content)
    
    def get_parsed_data(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        return self.parsed_data
    
    def save_to_jsonl(self, output_path: str):
        with open(output_path, "a", encoding="utf-8") as f:
            json.dump(self.parsed_data, f, ensure_ascii=False)
            f.write("\n")
            
    def sections_to_lists(self, document: dict = None, files_path: Union[str, List[str]] = None) -> List[str]:
           contents = []
           
           if document:
               contents = [section["content"] for section in document.get("sections", [])]
           elif files_path:
               if isinstance(files_path, str):
                   files_path = [files_path]
               
               for path in files_path:
                   with open(path, "r", encoding="utf-8") as f:
                       for line in f:
                           data = json.loads(line)
                           contents.extend([section["content"] for section in data.get("sections", [])])
           
           return contents
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        