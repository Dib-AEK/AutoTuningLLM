# AutoTuningLLM: Framework for Customized Text Generation and Fine-tuning LLM Models from PDF Documents using RAG (Retrieval-Augmented Generation)


This repository serves as a foundation for a framework that enables users to create customized Large Language Model (LLM) models by fine-tuning them on their own documents in PDF format using Retrieval-Augmented Generation (RAG).

## Overview

The primary goal of this project is to develop a versatile and user-friendly platform that uses LLMs fine-tuned on personal data to generate high-quality text. The framework will provide a suite of tools for:

*   **PDF Document Analysis**: Converting PDF files into text using Meta's Nougat model
*   **Context Retrieval**: Utilizing Meta's FAISS library for efficient similarity search and context retrieval
*   **Text Embeddings**: Employing the "jina-embeddings-v2-base-en" model for generating dense vector representations of text
*   **Model Fine-tuning**: Adjusting LLM models to optimize performance on specific document collections

## Dependencies

This project relies on the following dependencies:

*   `torch`
*   `transformers`
*   `faiss-gpu`
*   `sentence-transformers`
*   `jina`
*   `pdf2image`
*   `pymupdf`
*   `python-Levenshtein`
*   `nltk`
*   `ollama`

## Up-to-Now Status

Up to this point, we have successfully implemented the initial components of the framework: PDF-to-Text conversion using Nougat, context retrieval using FAISS, and text embeddings using "jina-embeddings-v2-base-en". The implementation of text post-processing, model leading, GUI, and fine-tuning is still in progress. 

## Future Development

In the coming phases, we plan to:

*   Implement advanced post-processing techniques to enhance text quality
*   Develop a model leading system that enables users to easily fine-tune their preferred open models
*   Establish a robust fine-tuning pipeline to optimize performance on diverse document collections

## Installation

To install the required dependencies, run the following command:
```bash
pip install torch transformers faiss-gpu sentence-transformers jina pdf2image pymupdf python-Levenshtein nltk ollama
```
Note: This is not an exhaustive list of dependencies.