# AutoTuningLLM: Framework for Customized Text Generation and Fine-tuning LLM Models from PDF Documents

This repository serves as a foundation for a framework that enables users to create customized Large Language Model (LLM) models by fine-tuning them on their own documents in PDF format.

## Overview

The primary goal of this project is to develop a versatile and user-friendly platform that leverages the power of LLMs fine tuned on personnal data to generate high-quality text. The framework will provide a suite of tools for:

*   **PDF Document Analysis**: Converting PDF files into text
*   **Model Fine-tuning**: Adjusting LLM models to optimize performance on specific document collections
*   **Customized Model Deployment**: Distributing and deploying the fine-tuned models

## Up-to-Now Status

Up to this point, we have successfully implemented the initial component of the framework: PDF-to-Text conversion. The `pdfToText.py` file contains the code for converting PDF files into text using the Nougat model.

```bash
# Install required libraries
pip install torch transformers datasets pdf2image pymupdf python-Levenshtein nltk

```

The implementation of text post-processing, model leading, and fine-tuning is still in progress. We will continue to expand upon this framework to make it an invaluable resource for researchers and developers seeking to harness the potential of LLMs.

## Future Development

In the coming phases, we plan to:

*   Implement advanced post-processing techniques to enhance text quality
*   Develop a model leading system that enables users to easly fine tune their prefered open models
*   Establish a robust fine-tuning pipeline to optimize performance on diverse document collections
