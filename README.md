# Research Paper Extractor — Groq Edition

## What's changed
- Groq API is the default AI provider path.
- The OpenAI Python package is removed from requirements and code.
- Local Ollama and Text-only modes remain available.
- Your custom columns are included: `device Propsed`, `Uses`, `Purpose`, `How is Work`.
- Excel includes Extracted Data, Evidence Review, Processing Log, and optional PDF Text sheets.

## Windows setup (CMD)
Open CMD in this folder and run:

```bat
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8502
```

In the app sidebar select Groq API, paste your Groq key into the password field, and test with one PDF first. Keep the model ID set to a model currently available in your Groq account. The default is `openai/gpt-oss-120b`; the previously discussed `llama-3.3-70b-versatile` may be retired/unavailable.

Never paste your API key into chat. Groq mode sends extracted PDF text to Groq. Verify the Evidence Review sheet.
