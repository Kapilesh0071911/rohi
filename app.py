from dotenv import load_dotenv
from openai import OpenAI
import json, os, requests, html as html_module
from pypdf import PdfReader
import gradio as gr
from fastapi import FastAPI
import uvicorn
from gradio.routes import mount_gradio_app

# Load environment variables
load_dotenv(override=True)
app = FastAPI()
port = int(os.environ.get("PORT", 4000))

# ---------- Notification + Tools ----------
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
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

# ---------- Me Class ----------
class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "CuraGuide"
        # Read LinkedIn pdf safely
        try:
            reader = PdfReader("me/linkedin.pdf")
            self.linkedin = "".join(page.extract_text() or "" for page in reader.pages)
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

    def handle_tool_call(self, tool_calls):
        results = []
        for call in tool_calls:
            tool_name = call.function.name
            args = json.loads(call.function.arguments)
            tool = globals().get(tool_name)
            result = tool(**args) if tool else {}
            results.append({"role":"tool", "content": json.dumps(result), "tool_call_id": call.id})
        return results

    def system_prompt(self):
        return (
            f"You are {self.name}, an AI-powered public health assistant. "
            "Provide disease awareness and health guidance. "
            "If you don't know an answer, use the record_unknown_question tool. "
            "If the user provides an email, record it using record_user_details.\n\n"
            f"## Summary:\n{self.summary}\n\n## LinkedIn:\n{self.linkedin}\n"
        )

    def chat(self, message, history):
        # ---- FLATTEN history to avoid nested lists ----
        flat_history = []
        for h in (history or []):
            if isinstance(h, list):
                flat_history.extend(h)
            else:
                flat_history.append(h)
        messages = [{"role":"system", "content": self.system_prompt()}] + flat_history + [
            {"role":"user", "content": message}
        ]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools
            )
            if response.choices[0].finish_reason == "tool_calls":
                message_part = response.choices[0].message
                tool_calls = message_part.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message_part)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content

me = Me()

# ---------- Render chat history to HTML ----------
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

# ---------- Custom CSS ----------
custom_css = r"""
body {
    background: linear-gradient(135deg, #7f9cf5 0%, #a78bfa 100%) !important;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    color: #282539;
}
#profile-sidebar {
    background: #322c50;
    color: #fff;
    min-width: 260px;
    max-width: 320px;
    padding: 24px;
    border-radius: 17px;
    box-shadow: 0 8px 32px rgba(56,18,169,0.20);
    margin-right: 26px;
}
#profile-sidebar h3 { color: #8bbafe; margin-bottom: 6px; }
#profile-sidebar .tagline { color: #e0e7ff; font-size: 13px; }
.placeholder-avatar {
    width: 82px; height: 82px;
    border-radius: 50%;
    background: #818cf8;
    display: flex; align-items: center; justify-content: center;
    color: #3730a3; font-weight: 700; font-size: 28px;
    margin-bottom: 16px;
    box-shadow: 0 0 0 4px #a78bfa50;
}
#logo {
    margin-bottom: 20px;
}
#chat-display { 
    background: #5b369f;  /* Purple shade */
    border-radius: 14px; 
    padding: 14px; 
    box-shadow: 0 6px 24px rgba(56,18,169,0.06);
    height: 540px;
    overflow: hidden;
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
    box-shadow: 0 3px 11px rgba(130, 90, 176, 0.08);
}
.bubble.user {
    align-self: flex-end;
    background: #6b52d8 !important;  /* Purple shade */
    color: #e0dfff !important;
}
.bubble.assistant {
    align-self: flex-start;
    background: #b9a6f8 !important;  /* Lighter purple */
    color: #3e2a9c !important;
}
.bubble.system {
    align-self: center;
    background: #d9d2fc;
    color: #543f99;
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
    border: 2.5px solid #6249d1 !important;
    resize: none !important;
    height: 48px !important;
    overflow-y: auto !important;
    background: #ebe9ff !important;
    color: #4b3cb1 !important;
}
#send-btn {
    background: linear-gradient(90deg,#6b52d8 60%,#a894fb 100%) !important;
    color: #fff !important;
    border-radius: 12px !important;
    font-weight: 600;
    letter-spacing: 0.04em;
    border: none;
}
#updates-display {
    background: #f0f4ff;
    border-radius: 14px;
    padding: 16px;
    height: 540px;
    overflow-y: auto;
    box-shadow: 0 6px 24px rgba(56,18,169,0.1);
    font-size: 15px;
    line-height: 1.4;
    color: #1e293b;
}
#updates-display a {
    color: #3b82f6;
    text-decoration: none;
}
#updates-display a:hover {
    text-decoration: underline;
}
.small-timestamp {
    color: #6a4bcf;
    font-size: 12px;
    font-style: italic;
}
"""

# ---------- Serper Integration ----------
SERPER_API_KEY = os.getenv("SERPER_API_KEY")  # Store your Serper API key in .env

def fetch_serper_updates():
    search_query = "latest disease outbreaks in India and health updates 2025"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    json_data = {
        "q": search_query,
        "gl": "in",
        "hl": "en"
    }
    response = requests.post("https://google.serper.dev/search", headers=headers, json=json_data)
    if response.status_code == 200:
        data = response.json()
        snippets = []
        for item in data.get("organic", [])[:5]:  # top 5 results
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            date = item.get("date") or item.get("published") or ""
            date_html = f"<br><span class='small-timestamp'>Updated: {date}</span>" if date else ""
            snippets.append(f"<b>{title}</b><br>{snippet}{date_html}<br><a href='{link}' target='_blank'>{link}</a><br><br>")
        return "".join(snippets)
    else:
        return "<b>Could not fetch live updates at this time.</b>"

# ---------- Gradio UI ----------
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row():
        with gr.Column(elem_id="profile-sidebar", scale=0.22):
            gr.Image("image.jpg", elem_id="logo", show_label=False, width=150)
            gr.Markdown("### CuraGuide\nAI-Powered Public Health Assistant")
            gr.Markdown("<div class='tagline'>Reliable disease awareness and guidance</div>")
        with gr.Column(scale=0.5):
            chat_display = gr.HTML(render_history_as_html([]), elem_id="chat-display")
            msg = gr.Textbox(
                placeholder="Type your complete question here...",
                elem_id="message-input",
                lines=1,
                max_lines=6,
                autofocus=True,
                show_label=False
            )
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
        with gr.Column(scale=0.28):
            gr.Markdown("### Latest Disease Outbreaks & Health Updates in India")
            updates_html = gr.HTML("<i>Loading updates...</i>", elem_id="updates-display")

            def load_updates():
                return fetch_serper_updates()

            demo.load(load_updates, outputs=updates_html)

# Serve Gradio app at "/"
app = mount_gradio_app(app, demo, path="/")

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
