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
    def __init__(self, markdown_text: str):
        self.markdown_text = markdown_text
        self.parsed_data = {
            "title": "",
            "authors": [],
            "abstract": "",
            "sections": []
        }
        self._parse_markdown()

    def _parse_markdown(self):
        html = markdown.markdown(self.markdown_text)
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        first_header = soup.find("h1")
        if first_header:
            self.parsed_data["title"] = first_header.text.strip()
            first_header.extract()

        # Extract authors
        self._extract_authors(soup)

        # Extract abstract
        abstract_header = soup.find("h6")
        if abstract_header and abstract_header.text.strip().lower() == "abstract":
            abstract_paragraph = abstract_header.find_next("p")
            if abstract_paragraph:
                self.parsed_data["abstract"] = abstract_paragraph.text.strip()
                abstract_paragraph.extract()

        # Extract sections
        self._extract_sections(soup)

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
    
    def _extract_sections(self, soup):
        current_section = {"heading": "", "content": ""}
        potential_section_headings = set()  # To keep track of potential section headings
    
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
                    strong_tag.extract()  # Remove the strong tag from the paragraph content
                else:
                    # Check if the paragraph text could be a section heading
                    text = tag.text.strip()
                    if self._is_potential_section_heading(text):
                        if current_section["heading"] and not self._contains_ignored_content(current_section["content"]):
                            self.parsed_data["sections"].append(current_section)
                        current_section = {"heading": text, "content": ""}
                        potential_section_headings.add(text)
                    else:
                        current_section["content"] += text + "\n"
            else:
                current_section["content"] += tag.text.strip() + "\n"
    
        if current_section["heading"] and not self._contains_ignored_content(current_section["content"]):
            self.parsed_data["sections"].append(current_section)
    
    def _is_potential_section_heading(self, text):
        # Heuristics to identify potential section headings
        # This is a simple example; you might need to adjust based on your data
        return len(text.split()) <= 5 and text.istitle()
    
    def _contains_ignored_content(self, content):
        # Check for "@", "orcid", or phone numbers in the content
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'  # Simple phone number pattern
        return "@" in content or "orcid" in content.lower() or re.search(phone_pattern, content)

    def get_parsed_data(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        return self.parsed_data

    def save_to_jsonl(self, output_path: str):
        with open(output_path, "a", encoding="utf-8") as f:
            json.dump(self.parsed_data, f, ensure_ascii=False)
            f.write("\n")

