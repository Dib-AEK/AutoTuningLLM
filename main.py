# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:26:05 2025

@author: abdel
"""

import os
from PdfToText.PdfToText import PdfToText
from PdfToText.NougatParser import NougatParser
from RAG.RAG import Embeddings, ContextRetriever, ContextManager
import GUI.functions as F

import gradio as gr
from pathlib import Path
import json
from tqdm import tqdm
import time


run_gui = True
# Define the paths
pdf_folder = Path(os.getcwd()+"/pdfDocuments")
cache_dir = Path(os.getcwd()+"/cache")
output_path_jsonl = cache_dir / f"pdfToTextCache/documents.jsonl"

# Get all PDF files in the folder
pdf_files = [pdf_folder/f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Convert PDFs to text and parse them
pdfToText = PdfToText()
parser = NougatParser(output_path_jsonl)

for file in pdf_files: #tqdm(pdf_files, desc="Converting pdf to text "):
    if not parser.is_in_parsed_files(file):
        generated_text = pdfToText.convert_by_batch(file, batch_size=2)
        parser.parse_document(markdown_text = generated_text, 
                              input_path    = str(file))
del pdfToText
time.sleep(1)


#Get all contents in one list
contents = parser.sections_to_lists(files_path=output_path_jsonl)
del parser
time.sleep(1)


# Create an Embeddings object to store the parsed documents
cache_dir_embeddings = cache_dir / "embeddings"
context_manager = ContextManager(documents = contents, overwrite = False, metric = "cosine",
                                  cache_dir = cache_dir_embeddings, embeddings_batch_size=2,
                                  device="cpu")


# Enrich the question  (doesn't work well)
def enrich_question(model:str, question:str):
    prompt = f"""You are an advanced AI assistant that enhances user queries to improve information retrieval. 

    Your task is to take the given question and expand it into a more detailed and representative version that captures its full meaning. 
    
    ### Instructions:
    1. Maintain the original intent of the question.
    2. Include relevant keywords, synonyms, or related concepts to make it more comprehensive.
    3. Preserve neutrality and avoid adding assumptions.
    4. Do NOT provide an answer—only return the enriched question.
    
    MOst importantly, only provide the new question, without explanation.
    ### User Question:
    {question}
    """
    return F.simple_chat_with_model(model, prompt)


# Use the context to ask questions to the LLM
def get_message(question, use_context: bool = False, model:str=None, max_number_of_caracters=8000):
    if use_context:
        # question = enrich_question(model, question)
        context = context_manager.retrieve_documents(question)
        context = "".join(context)
        context = context[:max_number_of_caracters]        
        if not context:  # Handle case where no relevant context is found
            return f"You don't have enough context to answer the question. Answer based on general knowledge.\n\nQuestion: {question}"
        
        message = (
            f"Use the following context to answer the question accurately. "
            f"If the context does not contain the answer, say 'I don't know' instead of guessing."
            f"Provide a clear and direct response without referencing this instruction.\n"
            f"### Context:\n{context}\n"
            f"### Question:\n{question}"
        )
    else:
        message = question  # Direct question without context
    
    return message



# Graphical user interface and communication with the model

models = F.get_available_models()
SYSTEM_MESSAGE = """You are an expert AI assistant with access to retrieved documents. 
When context is provided, follow these rules:
1. **Prioritize the given context** – If the retrieved information contains the answer, use it directly.
2. **Avoid making up information** – If the context does not contain the answer, say 'I don’t know' instead of guessing.
3. **Summarize and rephrase** – Provide clear and concise responses instead of copying large text chunks.
4. **Maintain neutrality and objectivity** – Base your answers strictly on the provided information.
5. **Cite context when necessary** – If the user asks for sources, refer to the relevant part of the retrieved documents without adding external assumptions.

If no context is provided, answer based on your general knowledge.

Always ensure accuracy and avoid speculation. Respond concisely and professionally.
"""

if run_gui:
    with gr.Blocks() as demo:
        
        gr.Markdown("# Ollama Chatbot Interface")
        
        with gr.Row():
            with gr.Column(scale=2):
                model_selector = gr.Dropdown(choices=models, 
                                             interactive=True,
                                             value="llama3.2:1b" if "llama3.2:1b" in models else models[0],
                                             label="Select Model")
                history = gr.Button("Reset History", variant="secondary")
                
                # SWITCH BUTTON for Context Toggle
                context_switch = gr.Checkbox(label="Enable Context", value=False)
                stop_button = gr.Button("Stop", variant="secondary")
            
            with gr.Column(scale=8):
                chatbot = gr.Chatbot()
                msg = gr.Textbox(label="Your message", placeholder="Type your message here...", lines=3)
                submit = gr.Button("Send", variant="secondary", size="sm")
    
        # Function to handle user input
        def user(user_message, history: list):
            history.append((user_message, ""))
            return "", history
    
        def user_interaction(model, chat_history, context_enabled):
            # Add context to the last message if necessary:
            user_message = None
            if context_enabled:
                user_message, _ = chat_history[-1]
                current_message = get_message(user_message, use_context = context_enabled, model=model)
                chat_history[-1] = (current_message, "")
            
            chat_history = F.chat_with_model(model, 
                                             chat_history, 
                                             impose_user_input_in_output=user_message,
                                             system_message=SYSTEM_MESSAGE)
            for update in chat_history:
                yield update
    
        def reset_chat():
            return []
    
        # Attach functions to components
        msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot]).then(
                   user_interaction, inputs=[model_selector, chatbot, context_switch], outputs=chatbot)
        
        submit.click(user_interaction, inputs=[model_selector, chatbot], outputs=chatbot)
        history.click(reset_chat, outputs=chatbot)
        stop_button.click(lambda: demo.close())
    
    # Launch the app
    demo.launch(inbrowser=True)