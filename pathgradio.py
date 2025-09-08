# pathgradio.py
# Gradio-based Customer Service Chatbot

import gradio as gr
from transformers import pipeline

# Load Turkish intent classification model
classifier = pipeline("sentiment-analysis", model="dbmdz/bert-base-turkish-cased")

def chatbot_response(user_input):
    result = classifier(user_input)[0]
    return f"Intent: {result['label']} (confidence: {result['score']:.2f})"

demo = gr.Interface(fn=chatbot_response, 
                    inputs="text", 
                    outputs="text",
                    title="Turkish Customer Service Chatbot",
                    description="A simple BERT-based intent classification chatbot.")

if __name__ == "__main__":
    demo.launch()
