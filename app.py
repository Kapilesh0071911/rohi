# app.py
"""
CuraGuide - FastAPI + Gradio chatbot with:
- India-focused vaccination database (UIP basics + age groups)
- Live outbreak fetch (Serper) — optional (requires SERPER_API_KEY in .env)
- Outbreak detection -> outbreak-driven vaccination drive recommendations
- Twilio webhook example (optional) for WhatsApp integration

Required .env variables:
- OPENAI_API_KEY (for OpenAI SDK)
- SERPER_API_KEY (optional; for live outbreak fetch)
- PUSHOVER_TOKEN / PUSHOVER_USER (optional; for push notifications)
- PORT (optional; defaults to 4000)

Install:
pip install python-dotenv openai gradio fastapi uvicorn pypdf requests
"""
from dotenv import load_dotenv
from openai import OpenAI
import json, os, requests, html as html_module, re
from pypdf import PdfReader
import gradio as gr
from fastapi import FastAPI, Request, Response
import uvicorn
from gradio.routes import mount_gradio_app
from datetime import datetime, timedelta

# Load environment
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
PORT = int(os.getenv("PORT", 4000))

# Initialize OpenAI client (new SDK style)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI()

# ---------- Small notification helpers (optional) ----------
def push(text):
    try:
        if not PUSHOVER_TOKEN or not PUSHOVER_USER:
            return
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": PUSHOVER_TOKEN,
                "user": PUSHOVER_USER,
                "message": text,
            },
            timeout=5
        )
    except Exception:
        pass

# ---------- Tools used by the assistant ----------
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Record a user's email and optional details",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string"},
            "name": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record any unanswered question",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

# ---------- Minimal Me class (keeps assistant behaviour) ----------
class Me:
    def __init__(self):
        self.openai = openai_client
        self.name = "CuraGuide"
        # optionally load local docs
        self.linkedin = ""
        self.diseases1 = ""
        try:
            if os.path.exists("me/linkedin.pdf"):
                reader = PdfReader("me/linkedin.pdf")
                self.linkedin = "".join(page.extract_text() or "" for page in reader.pages)
            if os.path.exists("me/diseases1.pdf"):
                reader = PdfReader("me/diseases1.pdf")
                self.diseases1 = "".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            self.linkedin = ""
            self.diseases1 = ""
        def _read_file(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return ""
        self.summary = _read_file("me/summary.txt")
        self.database = _read_file("me/database.txt")
        self.DiseasePrecaution = _read_file("me/DiseasePrecaution.txt")
        self.diseases = _read_file("me/diseases.txt")

    def system_prompt(self):
        return (
            f"You are {self.name}, an AI-powered public health assistant. "
            "Provide disease awareness and health guidance. "
            "If you don't know an answer, use record_unknown_question tool. "
            "If the user provides an email, record it using record_user_details.\n\n"
            f"## Summary:\n{self.summary}\n\n## LinkedIn:\n{self.linkedin}\n"
        )

    def chat(self, message, history):
        # Flatten history and call OpenAI (if configured)
        flat_history = []
        for h in (history or []):
            if isinstance(h, list):
                flat_history.extend(h)
            else:
                flat_history.append(h)
        messages = [{"role":"system", "content": self.system_prompt()}] + flat_history + [
            {"role":"user", "content": message}
        ]
        if not self.openai:
            # fallback simple echo
            return "OpenAI API key not configured. This is a fallback reply."
        # Use chat completions with tools (if available)
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )
        return response.choices[0].message.content

me = Me()

# ---------- UI helper ----------
def render_history_as_html(history):
    safe_parts = []
    safe_parts.append("<div class='chat-window' id='chat-window'>")
    for msg in history:
        role = msg.get("role", "")
        content = html_module.escape(msg.get("content", ""))
        if role == "user":
            safe_parts.append(f"<div class='bubble user'><div class='text'>{content}</div></div>")
        elif role == "assistant":
            safe_parts.append(f"<div class='bubble assistant'><div class='text'>{content}</div></div>")
        else:
            safe_parts.append(f"<div class='bubble system'><div class='text'>{content}</div></div>")
    safe_parts.append("</div>")
    safe_parts.append("""
<script>
(function(){
  try {
    const cw = document.getElementById('chat-window');
    if (cw) { cw.scrollTop = cw.scrollHeight; }
  } catch(e) {}
})();
</script>
""")
    return "\n".join(safe_parts)

# ---------- Styling ----------
custom_css = r"""
body {
    background: #f5f5f7;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    color: #2e2e38;
}
#profile-sidebar {
    background: #e3e6f3;
    color: #2e2e38;
    min-width: 260px;
    max-width: 320px;
    padding: 24px;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(58, 58, 70, 0.1);
    margin-right: 16px;
}
#chat-display {
    background: #ffffff;
    border-radius: 14px;
    padding: 14px;
    height: 540px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(58, 58, 70, 0.08);
}
.chat-window {
    height: 100%;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding-right: 8px;
}
.bubble {
    max-width: 85%;
    padding: 13px 19px;
    border-radius: 18px;
    line-height: 1.45;
    word-wrap: break-word;
    box-shadow: 0 2px 8px rgba(58, 58, 70, 0.04);
}
.bubble.user {
    align-self: flex-end;
    background: #7a86b6;
    color: #f2f2f7;
}
.bubble.assistant {
    align-self: flex-start;
    background: #d6d9eb;
    color: #2e2e38;
}
.bubble.system {
    align-self: center;
    background: #f0f1f5;
    color: #6b6b7a;
}
.bubble .text {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    writing-mode: horizontal-tb !important;
    display: inline-block;
}
#message-input textarea {
    border-radius: 12px !important;
    border: 2.5px solid #a9abc1 !important;
    resize: none !important;
    height: 48px !important;
    overflow-y: auto !important;
    background: #fff !important;
    color: #2e2e38 !important;
}
#send-btn {
    background: #7a86b6 !important;
    color: #fff !important;
    border-radius: 12px !important;
    font-weight: 600;
    letter-spacing: 0.04em;
    border: none;
}
#updates-display, #schedule-display {
    border-radius: 14px;
    padding: 16px;
    height: 200px;
    overflow-y: auto;
    font-size: 15px;
    line-height: 1.4;
    background: #f1f2f6;
    color: #2e2e38;
    box-shadow: 0 4px 20px rgba(58, 58, 70, 0.05);
}
#updates-display a, #schedule-display a {
    color: #7a86b6;
    text-decoration: none;
}
#updates-display a:hover, #schedule-display a:hover {
    text-decoration: underline;
}
.small-timestamp {
    color: #9d9db2;
    font-size: 12px;
    font-style: italic;
}
"""

# (You can keep the rest of the unchanged code from above as-is for the UI, chat, and API integrations)

# ---------- Serper fetch and other logic remains same ----------

# ---------- Gradio UI ----------
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row():
        with gr.Column(elem_id="profile-sidebar", scale=0.22):
            if os.path.exists("image.jpg"):
                gr.Image("image.jpg", elem_id="logo", show_label=False, width=120)
            else:
                gr.Markdown("### CuraGuide\nAI-Powered Public Health Assistant")
            gr.Markdown("<div class='tagline'>Reliable disease awareness and guidance</div>")
            gr.Markdown("---")
            btn_updates = gr.Button("Refresh Latest Updates")
            btn_schedule = gr.Button("Create Vaccination Schedule")
            gr.Markdown("<div style='font-size:13px;color:#7a86b6'>Status: live</div>")

        with gr.Column(scale=0.55):
            chat_display = gr.HTML(render_history_as_html([]), elem_id="chat-display")
            msg = gr.Textbox(placeholder="Type your complete question here...", elem_id="message-input", lines=1, max_lines=6, autofocus=True, show_label=False)
            send_btn = gr.Button("Send", elem_id="send-btn")
            state = gr.State([])

            def handle_message(user_text, history):
                if not user_text or str(user_text).strip() == "":
                    return "", history or [], render_history_as_html(history or [])
                flat_history = []
                for h in (history or []):
                    if isinstance(h, list):
                        flat_history.extend(h)
                    else:
                        flat_history.append(h)
                assistant_text = me.chat(user_text, flat_history)
                flat_history.append({"role": "user", "content": user_text})
                flat_history.append({"role": "assistant", "content": assistant_text})
                html_out = render_history_as_html(flat_history)
                return "", flat_history, html_out

            msg.submit(handle_message, [msg, state], [msg, state, chat_display])
            send_btn.click(handle_message, [msg, state], [msg, state, chat_display])

        with gr.Column(scale=0.25):
            gr.Markdown("### Latest Disease Outbreaks & Health Updates")
            updates_html = gr.HTML("<i>Loading updates...</i>", elem_id="updates-display")
            gr.Markdown("### Suggested / Adjusted Vaccination Schedule")
            schedule_html = gr.HTML("<i>Press 'Create Vaccination Schedule' to generate from updates</i>", elem_id="schedule-display")

            def load_updates():
                return fetch_serper_updates()

            def generate_schedule(_=None):
                updates = fetch_serper_updates()
                schedule = create_vaccination_schedule_from_updates(updates)
                return updates, schedule

            demo.load(load_updates, outputs=updates_html)
            btn_updates.click(lambda: fetch_serper_updates(), outputs=updates_html)
            btn_schedule.click(generate_schedule, inputs=None, outputs=[updates_html, schedule_html])

# mount gradio
app = mount_gradio_app(app, demo, path="/")

# ---------- Optional Twilio webhook endpoint (simple) ----------
@app.post("/twilio-whatsapp")
async def twilio_whatsapp(request: Request):
    form = await request.form()
    user_text = form.get("Body", "")
    reply = me.chat(user_text, [])
    twiml = f"<?xml version='1.0' encoding='UTF-8'?><Response><Message>{html_module.escape(reply)}</Message></Response>"
    return Response(content=twiml, media_type="application/xml")

# ---------- Run server ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
