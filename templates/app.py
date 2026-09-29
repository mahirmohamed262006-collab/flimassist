import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

from chatbot_config import SYSTEM_PROMPT

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

client = genai.Client(api_key=API_KEY) if API_KEY else None


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please enter a message."}), 400

    if len(message) > 4000:
        return jsonify({"error": "Please keep your message under 4,000 characters."}), 400

    if client is None:
        return jsonify({
            "error": "Gemini API key is not configured. Add your key to the .env file."
        }), 503

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message.strip(),
            config={"system_instruction": SYSTEM_PROMPT},
        )
        answer = (response.text or "").strip()
        if not answer:
            answer = "I couldn't create a response just now. Please try again."
        return jsonify({"reply": answer})
    except Exception:
        app.logger.exception("Gemini API request failed")
        return jsonify({
            "error": "The assistant is temporarily unavailable. Please try again shortly."
        }), 502


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
