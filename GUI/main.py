# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 20:07:12 2025

@author: abdel
"""

import gradio as gr
import functions as F


models = F.get_available_models()

with gr.Blocks() as demo:
    
    gr.Markdown("# Ollama Chatbot Interface")
    with gr.Row():
        with gr.Column(scale=2):
            model_selector = gr.Dropdown(choices=models, 
                                         interactive=True,
                                         value="llama3.2:1b",
                                         label="Select Model")
            history = gr.Button("Reset History", variant="secondary")
            stop_button = gr.Button("Stop", variant="secondary")
        
        with gr.Column(scale=8):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Your message", placeholder="Type your message here...", lines=3)
            submit = gr.Button("Send", variant="secondary", size="sm")
            
    def user(user_message, history: list):
        history.append((user_message,""))
        return "", history
    
    def user_interaction(model, chat_history):
        chat_history = F.chat_with_model(model, chat_history)
        for update in chat_history:
            yield update
    
    def reset_chat():
        return []
 
    msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot]).then(
               user_interaction, inputs=[model_selector, chatbot], outputs=chatbot)
    
    # msg.submit(user, inputs=[model_selector, msg, chatbot], outputs=[chatbot, msg])
    submit.click(user_interaction, inputs=[model_selector, msg, chatbot], outputs=[chatbot, msg])
    history.click(reset_chat, outputs=chatbot)
    stop_button.click(lambda: demo.close())

# Launch the app
demo.launch()




