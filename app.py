from dotenv import load_dotenv
from openai import OpenAI
import json, os, requests
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
        self.name = "kapilesh"
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = "".join(page.extract_text() or "" for page in reader.pages)
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()
        with open("me/database.txt", "r", encoding="utf-8") as f:
            self.database = f.read()
        with open("me/DiseasePrecaution.txt", "r", encoding="utf-8") as f:
            self.DiseasePrecaution = f.read()
        with open("me/diseases.txt", "r", encoding="utf-8") as f:
            self.diseases = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for call in tool_calls:
            tool_name = call.function.name
            args = json.loads(call.function.arguments)
            tool = globals().get(tool_name)
            result = tool(**args) if tool else {}
            results.append({"role": "tool",
                            "content": json.dumps(result),
                            "tool_call_id": call.id})
        return results

    def system_prompt(self):
        return (
            f"You are {self.name}, an AI-powered public health assistant. "
            f"Provide disease awareness and health guidance. "
            f"If unsure, record the question using record_unknown_question.\n\n"
            f"## Summary:\n{self.summary}\n\n## LinkedIn:\n{self.linkedin}\n"
        )

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            resp = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools
            )
            if resp.choices[0].finish_reason == "tool_calls":
                msg = resp.choices[0].message
                results = self.handle_tool_call(msg.tool_calls)
                messages.append(msg)
                messages.extend(results)
            else:
                done = True
        return resp.choices[0].message.content

me = Me()

# ---------- High-Contrast Medical Theme ----------
custom_css = """
body {
    background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%) !important;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    color: #1e293b;
}

/* Sidebar */
#profile-sidebar {
    background: #ffffff;
    color: #1e293b;
    min-width: 260px;
    max-width: 320px;
    padding: 28px 22px;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    margin-right: 24px;
}
#profile-sidebar h3 {
    color: #1d4ed8;
    margin-bottom: 8px;
}
#profile-sidebar .tagline {
    color: #475569;
    font-size: 14px;
}
.placeholder-avatar {
    width: 82px; height: 82px;
    border-radius: 50%;
    background: #93c5fd;
    display: flex; align-items: center; justify-content: center;
    color: #1e3a8a;
    font-weight: bold; font-size: 28px;
    margin-bottom: 18px;
}

/* Chat Area */
.gr-chatbot {
    background: #ffffff !important;
    border-radius: 18px;
    padding: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
    width: 100% !important;
}

/* Chat Messages – FIX for sentence display */
.message {
    border-radius: 16px;
    padding: 13px 20px;
    margin: 10px 0;
    max-width: 85%;
    line-height: 1.5;
    white-space: pre-wrap;    /* ✅ keeps sentences intact */
    word-break: break-word;   /* ✅ breaks only long words if needed */
    overflow-wrap: break-word;
}
.user, .assistant {
    display: inline-block;    /* ✅ prevents full-width stretching */
}

/* User & Assistant Colors */
.user {
    background: #bfdbfe !important;
    color: #1e3a8a !important;
    align-self: flex-end;
}
.assistant {
    background: #d1fae5 !important;
    color: #065f46 !important;
    align-self: flex-start;
}

/* Input Box */
#message-input textarea {
    border-radius: 12px !important;
    border: 2px solid #93c5fd !important;
    resize: none !important;
    height: 48px !important;
    overflow-y: auto !important;
    white-space: normal !important; /* ✅ ensures normal typing */
}

/* Send Button */
#send-btn {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    font-weight: 600;
}
"""

# ---------- Gradio UI ----------
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row():
        with gr.Column(elem_id="profile-sidebar", scale=0.35):
            gr.HTML("<div class='placeholder-avatar'>K</div>")
            gr.Markdown("### Kapilesh\nAI-Powered Public Health Assistant")
            gr.Markdown("<div class='tagline'>Reliable disease awareness and guidance</div>")
        with gr.Column():
            chatbot = gr.Chatbot(label="Medical Chat", elem_classes=["gr-chatbot"], height=540)
            msg = gr.Textbox(
                placeholder="Type your complete question here...",
                elem_id="message-input",
                scale=8,
                autofocus=True,
                lines=1,       # single-line typing
                max_lines=3    # allow limited multi-line if Shift+Enter is used
            )
            send_btn = gr.Button("Send", elem_id="send-btn", scale=1)

            def handle_message(message, chat_history):
                answer = me.chat(message, chat_history or [])
                chat_history.append((message, answer))
                return "", chat_history

            msg.submit(handle_message, [msg, chatbot], [msg, chatbot])
            send_btn.click(handle_message, [msg, chatbot], [msg, chatbot])

# Mount the Gradio app
app = mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
