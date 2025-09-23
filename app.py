"""
CuraGuide - FastAPI + Gradio chatbot with:
- India-focused vaccination database (UIP basics + age groups)
- Live outbreak fetch (Serper) — optional (requires SERPER_API_KEY in .env)
- Outbreak detection -> outbreak-driven vaccination drive recommendations
- Twilio webhook example (optional) for SMS integration

Required .env variables:
- OPENAI_API_KEY (for OpenAI SDK)
- SERPER_API_KEY (optional; for live outbreak fetch)
- PUSHOVER_TOKEN / PUSHOVER_USER (optional; for push notifications)
- TWILIO_ACCOUNT_SID (for Twilio)
- TWILIO_AUTH_TOKEN (for Twilio)
- TWILIO_PHONE_NUMBER (for Twilio)
- PORT (optional; defaults to 4000)

Install dependencies:
pip install python-dotenv openai gradio fastapi uvicorn pypdf requests twilio
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
from twilio.rest import Client

# Load environment variables
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
PORT = int(os.getenv("PORT", 4000))
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Setup OpenAI client
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Setup Twilio client
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()

# ---------- Notification helpers ----------
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

# ---------- Twilio SMS sending function ----------
def send_sms(to_number, message):
    if not twilio_client:
        print("Twilio client not configured.")
        return
    twilio_client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=to_number
    )

# ---------- Tools ----------
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

# ---------- Me class ----------
class Me:
    def __init__(self):
        self.openai = openai_client
        self.name = "CuraGuide"
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
            return "OpenAI API key not configured. This is a fallback reply."
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )
        return response.choices[0].message.content

me = Me()

# ---------- Helpers ----------
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

# ---------- Serper live outbreak info ----------
def fetch_serper_updates(search_query=None):
    if not search_query:
        search_query = "latest disease outbreaks India 2025 health updates"
    if not SERPER_API_KEY:
        return "<b>Serper API key not found (SERPER_API_KEY in .env). Cannot fetch live updates.</b>"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    json_data = {"q": search_query, "gl": "in", "hl": "en"}
    try:
        resp = requests.post("https://google.serper.dev/search", headers=headers, json=json_data, timeout=12)
    except Exception as e:
        return f"<b>Could not fetch live updates: {html_module.escape(str(e))}</b>"
    if resp.status_code != 200:
        return f"<b>Could not fetch live updates (status {resp.status_code}).</b>"
    data = resp.json()
    snippets = []
    for item in data.get("organic", [])[:8]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        date = item.get("date") or item.get("published") or ""
        date_html = f"<br><span class='small-timestamp'>Updated: {date}</span>" if date else ""
        snippets.append(f"<b>{html_module.escape(title)}</b><br>{html_module.escape(snippet)}{date_html}<br><a href='{link}' target='_blank'>{link}</a><br><br>")
    if not snippets:
        return "<b>No updates found.</b>"
    return "".join(snippets)

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

/* Light mode base colors - text is dark, backgrounds light */
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

/* Dark mode overrides */
@media (prefers-color-scheme: dark) {
    body {
        background: #121212;
        color: #e0e0e0;
    }
    #profile-sidebar {
        background: #262626;
        color: #e0e0e0;
    }
    #chat-display {
        background: #1e1e1e;
        color: #e0e0e0;
    }
    .bubble.user {
        background: #5e55a3;
        color: #fefefe;
    }
    .bubble.assistant {
        background: #34354f;
        color: #e0e0e0;
    }
    .bubble.system {
        background: #2f2f3f;
        color: #bbbbbb;
    }
    #updates-display, #schedule-display {
        background: #2b2b3b;
        color: #dedede;
    }
    #updates-display a, #schedule-display a {
        color: #a3a0f9;
    }
    #updates-display a:hover, #schedule-display a:hover {
        text-decoration: underline;
    }
    #message-input textarea {
        background: #252525 !important;
        color: #e0e0e0 !important;
        border-color: #6a6a8a !important;
    }
    #send-btn {
        background: linear-gradient(90deg, #494377 60%, #5f59b8 100%) !important;
        color: #e0e0e0 !important;
    }
}
"""

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

# ---------- Twilio SMS webhook for receiving messages ----------
@app.post("/twilio-sms")
async def twilio_sms_webhook(request: Request):
    form = await request.form()
    from_number = form.get("From")
    to_number = form.get("To")
    body = form.get("Body")

    print(f"Received SMS from {from_number}: {body}")

    # Pass received message to chatbot
    reply = me.chat(body, [])

    # Response TwiML to reply via SMS
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response><Message>{html_module.escape(reply)}</Message></Response>"""

    return Response(content=twiml_response, media_type="application/xml")

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)