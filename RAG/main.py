from RAG import Embeddings, ContextRetriever
import os
import pickle
import json
import numpy as np
import faiss
from transformers import AutoModel
import sys

embeddings = Embeddings()
context_retriever = ContextRetriever(embeddings)