from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from fastapi import FastAPI
import uvicorn
from gradio.routes import mount_gradio_app

# Load environment variables
load_dotenv(override=True)

# FastAPI app
app = FastAPI()
port = int(os.environ.get("PORT", 4000))

# ---------- Notification + Tool Functions ----------
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
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            },
            "notes": {
                "type": "string",
                "description": "Any additional context from the conversation"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
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
        # Read LinkedIn PDF
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        # Read summary file
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
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, " \
                        f"and also answering questions on {self.name}'s database, diseases, and disease precautions. " \
                        f"Your responsibility is to represent {self.name} faithfully. If you don't know an answer, " \
                        f"use the record_unknown_question tool. If the user provides an email, record it using " \
                        f"record_user_details.\n\n"
        system_prompt += f"## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        return system_prompt

    # Main chat method for Gradio
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content

# ---------- Modern Gradio UI (Blocks) ----------
me = Me()

# --- Custom Sidebar and CSS for Modern Chat UI ---
custom_css = """
#profile-sidebar {
    background: #232334;
    color: #fff;
    min-width: 260px;
    max-width: 320px;
    padding: 32px 24px;
    border-radius: 14px;
    box-shadow: 2px 0 24px 0 rgba(0,0,0,0.14);
    margin-right: 24px;
}
#profile-sidebar img {
    border-radius: 50%;
    width: 82px;
    margin-bottom: 22px;
    border: 2px solid #33344a;
}
.gr-chatbot {
    background: #181825 !important;
    border-radius: 16px;
    padding: 0 10px 12px 10px;
}
.message {
    border-radius: 17px;
    padding: 13px 21px;
    margin: 11px 0;
    max-width: 76%;
}
.user {
    background: #44475a !important;
    color: #fafafc !important;
    align-self: flex-end;
}
.assistant {
    background: #35394d !important;
    color: #fafafc !important;
    align-self: flex-start;
}
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as demo:
    with gr.Row():
        with gr.Column(elem_id="profile-sidebar", scale=0.35):
            gr.Image("me/profile.png", show_label=False)  # <--- Your profile image, adjust path as needed
            gr.Markdown("### kapilesh\nAI/ML Engineer\nkapilesh@email.com")
            gr.Markdown("> Empathy | Innovation | Impact")
        with gr.Column():
            chatbot = gr.Chatbot(label="Chat with kapilesh", elem_classes=["gr-chatbot"], height=540)
            msg = gr.Textbox(placeholder="Type your message...", elem_id="message-input", scale=8, autofocus=True)
            send_btn = gr.Button("Send", elem_id="send-btn", scale=1)

            def handle_message(message, chat_history):
                answer = me.chat(message, chat_history)
                chat_history = chat_history or []
                chat_history.append((message, answer))
                return "", chat_history

            msg.submit(handle_message, [msg, chatbot], [msg, chatbot])
            send_btn.click(handle_message, [msg, chatbot], [msg, chatbot])

# Serve Gradio app at "/"
app = mount_gradio_app(app, demo, path="/")

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
