# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:16:18 2025

@author: abdel
"""


import os
import pickle
import json
import numpy as np
import faiss
from transformers import AutoModel
import sys
from typing import Union, List
import numpy as np


class Embeddings:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-en", 
                      cache_dir: str = "../cache",
                      overwrite: bool=True):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.overwrite = overwrite
        
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.max_seq_length = getattr(self.model.config, 'max_position_embeddings', 1024 * 4)
        self.embedding_dim = getattr(self.model.config, 'hidden_size', None)
        self._embeddings = None
        self.documents_path = self._cache_path("documents.jsonl")
        self.embeddings_path = self._cache_path("embeddings.pkl")

    def _cache_path(self, filename: str) -> str:
        """
        Constructs the full cache file path for a given filename based on the cache directory.

        Parameters:
        filename (str): The name of the file for which the cache path is to be constructed.

        Returns:
        str: The full path to the cache file within the cache directory.
        """

        return os.path.join(self.cache_dir, filename)

    def to_embeddings(self, documents: list):
        """
        Converts a list of documents to embeddings using the model's encode method.

        Parameters:
        documents (list): list of text documents to convert to embeddings.

        Returns:
        np.ndarray: 2D array of shape (len(documents), self.embedding_dim).
        """
        return self.model.encode(documents, max_length=self.max_seq_length)

    def add_documents(self, documents: list):
        """
        Add documents of given documents in a file in the cache directory.
        
        If the embeddings file already exists, it loads the existing embeddings and
        appends the new ones to it. If the total size of the embeddings exceeds 512MB,
        it does not store the embeddings in memory to avoid memory issues.
        
        Parameters:
        documents (list): list of text documents to store the embeddings of.
        """
        stored_documents = set()
        new_documents = []
        
        if self.overwrite:
            print("Overwrite is true, old documents (documents & embeddings) are removed!")
            if os.path.exists(self.documents_path):
                os.remove(self.documents_path)
            if os.path.exists(self.embeddings_path):
                os.remove(self.embeddings_path)
                
        
        
        if os.path.exists(self.documents_path):
            with open(self.documents_path, "r", encoding="utf-8") as doc_file:
                stored_documents = {json.loads(line)["text"] for line in doc_file}
        
        new_documents = [doc for doc in documents if doc not in stored_documents]
        
        if not new_documents:
            return  # No new documents to process
        
        new_embeddings = self.to_embeddings(new_documents)
        
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, "rb") as f:
                cached_data = pickle.load(f)
            existing_embeddings = cached_data["embeddings"]
            all_embeddings = np.vstack((existing_embeddings, new_embeddings))
        else:
            all_embeddings = new_embeddings
        
        with open(self.embeddings_path, "wb") as f:
            pickle.dump({"embeddings": all_embeddings}, f)

        memory_size_mb = sys.getsizeof(all_embeddings) / (1024 * 1024)
        size_limit = 512
        if memory_size_mb <= size_limit:
            self._embeddings = all_embeddings
        else:
            self._embeddings = None
            print(f"Embeddings are larger than {size_limit}MB. Avoid storing them in memory. Storing them in a file instead.")

        with open(self.documents_path, "a", encoding="utf-8") as doc_file:
            for doc in new_documents:
                json.dump({"text": doc}, doc_file)
                doc_file.write("\n")

    def get_documents(self, i: Union[int, List[int],str]):
        """
        Retrieve documents from the stored documents by index or indices.

        Args:
            i (int or List[int]): Index or indices of the documents to retrieve.

        Returns:
            str or List[str]: The document(s) at the specified index or indices.

        Raises:
            IndexError: If the index or indices are out of range.
        """
        with open(self.documents_path, "r", encoding="utf-8") as doc_file:
            lines = [json.loads(line)["text"] for line in doc_file]
            
        if isinstance(i, int):
            return lines[i]
        elif isinstance(i, str):
            assert i.lower() == "all", "Only 'all' is supported when using a string"
            return lines
        else:
            return [lines[idx] for idx in i]
                
        raise IndexError("Document index out of range.")

    @property
    def dimension(self):
        """
        Return the dimension of the embeddings.

        This is the size of each vector in the embeddings array.

        Raises:
            ValueError: If the embedding dimension is not set yet.
        """
        if self.embedding_dim is None:
            raise ValueError("Embedding dimension is not set yet.")
        return self.embedding_dim
    
    @property
    def embeddings(self):
        """
        Return the stored embeddings.

        If the embeddings are not stored in memory, try to load them from the cache file.
        If the cache file does not exist, print a warning and return None.

        Returns:
            The stored embeddings as a numpy array, or None if no embeddings are stored.
        """
        if self._embeddings is None:
            if os.path.exists(self.embeddings_path):
                embeddings = pickle.load(open(self.embeddings_path, "rb"))["embeddings"]
                return embeddings
            else:
                print("There are no embeddings stored")
                return None
        else:
            return self._embeddings

    def __len__(self):
        with open(self.documents_path, "r", encoding="utf-8") as doc_file:
            return sum(1 for _ in doc_file)



import os
import faiss
import numpy as np
from typing import Union, List

class ContextRetriever:
    def __init__(self, embeddings: Embeddings, index_path: str = "../cache/index.faiss", 
                 overwrite: bool = True, metric: str = "l2"):
        """
        ContextRetriever for retrieving similar documents based on embeddings.

        Args:
            embeddings (Embeddings): The embeddings object.
            index_path (str): Path to store FAISS index.
            overwrite (bool): Whether to overwrite existing FAISS index.
            metric (str): Distance metric, either 'l2' (default) or 'cosine'.
        """
        assert metric in ["l2", "cosine"], "metric must be 'l2' or 'cosine'"
        self.embeddings = embeddings
        self.index_path = index_path
        self.overwrite = overwrite
        self.metric = metric
        self.get_index_faiss()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """ Normalize vectors for cosine similarity. """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)  # Avoid division by zero

    def get_index_faiss(self):
        """ Load or create FAISS index based on selected metric. """
        if os.path.exists(self.index_path) and self.overwrite:
            print("Overwrite is true, old FAISS index is removed!")
            os.remove(self.index_path)

        if os.path.exists(self.index_path):
            self.index_faiss = faiss.read_index(self.index_path)
        else:
            embeddings_vector = self.embeddings.embeddings
            if embeddings_vector is not None:
                if self.metric == "cosine":
                    embeddings_vector = self._normalize(embeddings_vector)
                    self.index_faiss = faiss.IndexFlatIP(self.embeddings.dimension)  # Inner Product for Cosine
                else:
                    self.index_faiss = faiss.IndexFlatL2(self.embeddings.dimension)  # L2 Distance

                self.index_faiss.add(embeddings_vector)
                faiss.write_index(self.index_faiss, self.index_path)
            else:
                self.index_faiss = None

    def retrieve_documents(self, query: Union[str, List[str]], top_k: int = 2):
        """ Retrieve top-k similar documents for a given query. """
        if self.index_faiss is None:
            self.get_index_faiss()
        
        if isinstance(query, str):
            query = [query]
        
        query_embedding = self.embeddings.model.encode(query)
        if self.metric == "cosine":
            query_embedding = self._normalize(query_embedding)
            top_k = top_k*2

        dist, indices = self.index_faiss.search(np.array(query_embedding), top_k)
        dist, indices = dist[0], indices[0]
        
        if self.metric == "cosine":
            high_similarity = dist>0.5
            if high_similarity.sum()<=top_k/2:
                indices = indices[:2]
            if high_similarity.sum()>top_k/2:
                indices = indices[high_similarity]
                
        return self.embeddings.get_documents(indices)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
