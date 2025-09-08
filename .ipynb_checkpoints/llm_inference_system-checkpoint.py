import os
from openai import OpenAI
import gradio as gr
from langdetect import detect
from textblob import TextBlob
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

# 🔑 Setup OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# 🌍 Supported languages
LANGUAGES = {
    "en": "English",
    "tr": "Turkish",
    "de": "German",
    "fr": "French",
    "es": "Spanish"
}

# 🎭 Sentiment analysis (transformers pipeline)
sentiment_analyzer = pipeline("sentiment-analysis")

# 🎨 Story genres
GENRES = ["fantasy", "sci-fi", "adventure", "romance"]

# 📝 Generate story
def generate_story(prompt, genre, language):
    try:
        # Detect language if set to auto
        if language == "auto":
            language = detect(prompt)
            if language not in LANGUAGES:
                language = "en"

        # Build context
        genre_text = f"Write a {genre} story"
        user_prompt = f"{genre_text} in {LANGUAGES[language]}: {prompt}"

        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative multilingual story writer."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=500,
        )

        story = response.choices[0].message.content

        # Analyze sentiment
        sentiment = sentiment_analyzer(story[:512])[0]
        sentiment_text = f"Sentiment: {sentiment['label']} (Confidence: {sentiment['score']:.2f})"

        # Quality check (using TextBlob for simplicity)
        tb = TextBlob(story)
        quality = f"Readability Score (polarity): {tb.sentiment.polarity:.2f}"

        return story, sentiment_text, quality

    except Exception as e:
        return f"⚠️ Error: {str(e)}", "", ""

# 🌐 Gradio interface
with gr.Blocks(theme="default") as demo:
    gr.Markdown("# 🌍 Multilingual Story Generator\nCreate stories in English, Turkish, German, French, and Spanish!")

    with gr.Row():
        prompt = gr.Textbox(lines=3, label="Enter your story idea")
        genre = gr.Dropdown(GENRES, label="Select genre", value="fantasy")
        language = gr.Dropdown(["auto"] + list(LANGUAGES.keys()), label="Language", value="auto")

    generate_btn = gr.Button("✨ Generate Story")

    with gr.Row():
        story_output = gr.Textbox(label="Generated Story", lines=10)
    with gr.Row():
        sentiment_output = gr.Textbox(label="Sentiment Analysis")
        quality_output = gr.Textbox(label="Quality Check")

    generate_btn.click(
        fn=generate_story,
        inputs=[prompt, genre, language],
        outputs=[story_output, sentiment_output, quality_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
