# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:48:50 2025

@author: abdel
"""
import ollama




# transform history for printing in the GUI
def transform_history(pairs):
    if not len(pairs):
        return []
    
    history = []
    for user_msg, assistant_msg in pairs:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
    
    return history

def inverse_transform(history):
    role_map = {"user": [], "assistant": []}
    
    for message in history:
        role_map[message['role']].append(message["content"])
    
    return list(zip(role_map["user"], role_map["assistant"]))




# get the available models (models already installed)
def get_available_models():
    """
    Return a list of the names of the models already installed in this environment
    
    Returns
    -------
    list
        List of model names
    """
    return [x.model for x in ollama.list().models]




# Function to handle chat with the selected model
def chat_with_model(model_name, history):
    """
    Handles chat with the selected model
    
    Parameters
    ----------
    model_name : str
        The name of the model to use for the chat
    history : list
        The chat history to use as context for the chat
    
    Yields
    ------
    list
        The updated chat history
    """

    if not model_name:
        return "Please select a model first."
    
    transformed_history = transform_history(history)
    
    # Adding question to chat history
    transformed_history = transformed_history[:-1]
    
    # Generating response using Ollama
    stream  = ollama.chat(model=model_name, messages=transformed_history, stream=True)
    
    # Adding response to chat history
    user_input = history[-1][0]
    response = ""
    history[-1] = (user_input, response)
    
    for chunk in stream:
      part = chunk['message']['content']
      response = response + part
      history[-1] = (user_input, response)
      yield history
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
