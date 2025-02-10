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

class Embeddings:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-en", cache_dir: str = "../cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
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

    def store_embeddings(self, documents: list):
        """
        Stores the embeddings of given documents in a file in the cache directory.
        
        If the embeddings file already exists, it loads the existing embeddings and
        appends the new ones to it. If the total size of the embeddings exceeds 512MB,
        it does not store the embeddings in memory to avoid memory issues.
        
        Parameters:
        documents (list): list of text documents to store the embeddings of.
        """
        stored_documents = set()
        new_documents = []
        
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

    def get_documents(self, i: Union[int, List[int]]):
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
            if isinstance(i, int):
                for idx, line in enumerate(doc_file):
                    if idx == i:
                        return json.loads(line)["text"]
            else:
                documents = {}
                for idx, line in enumerate(doc_file):
                    if idx in i:
                        documents[idx] = json.loads(line)["text"]
                return [documents[idx] for idx in i] 
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
        else:
            return self._embeddings





class ContextRetriever:
    def __init__(self, embeddings: Embeddings, index_path: str = "../cache/index.faiss"):
        self.embeddings = embeddings
        self.index_path = index_path
        
        if os.path.exists(index_path):
            self.index_faiss = faiss.read_index(index_path)
        else:
            self.index_faiss = faiss.IndexFlatL2(embeddings.dimension)
            self.index_faiss.add(embeddings.embeddings)
            faiss.write_index(self.index_faiss, index_path)

    def retrieve_documents(self, query: str, top_k: int = 2):
        query_embedding = self.embeddings.model.encode([query])
        _, indices = self.index_faiss.search(np.array(query_embedding), top_k)
        return [self.embeddings.get_document(i) for i in indices[0]]
