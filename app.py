from dotenv import load_dotenv
from openai import OpenAI
import json
import os
from pypdf import PdfReader
import gradio as gr
from fastapi import FastAPI
import uvicorn
from gradio.routes import mount_gradio_app
import requests

load_dotenv(override=True)

PORT = int(os.getenv("PORT", 4000))

# Push Notification Helper
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

# Tool Implementations
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}

tools = [
    {
        "name": "record_user_details",
        "description": "Record user email, name, and notes",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "name": {"type": "string"},
                "notes": {"type": "string"}
            },
            "required": ["email"]
        }
    },
    {
        "name": "record_unknown_question",
        "description": "Log unknown questions",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"}
            },
            "required": ["question"]
        }
    },
    {
        "name": "web_search",
        "description": "Use this tool to search the web for up-to-date information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query string"}
            },
            "required": ["query"]
        }
    }
]

# ---- Me Class ----
class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "kapilesh"

        self.linkedin = ""
        reader = PdfReader("me/linkedin.pdf")
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def system_prompt(self):
        return f"""You are {self.name}, a personal assistant answering questions about {self.name}'s career, background, skills, and experience.
Summary:
{self.summary}

LinkedIn:
{self.linkedin}

Use web_search function to search the web when needed."""

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
                tool_call = response.choices[0].message.tool_calls[0]
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                if tool_name == "record_user_details":
                    result = record_user_details(**arguments)
                elif tool_name == "record_unknown_question":
                    result = record_unknown_question(**arguments)
                elif tool_name == "web_search":
                    query = arguments.get("query", "")
                    # Directly call OpenAI WebSearchTool by making another API call (optional)
                    search_response = self.openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": f"Search the web for: {query}"}]
                    )
                    result = {"search_result": search_response.choices[0].message.content}
                else:
                    result = {"error": "Unknown tool"}

                messages.append(response.choices[0].message)
                messages.append({"role": "tool", "content": json.dumps(result)})
            else:
                done = True

        return response.choices[0].message.content


# ---- Mount Gradio ----
me = Me()
demo = gr.ChatInterface(me.chat, type="messages")

app = FastAPI()
app = mount_gradio_app(app, demo, path="/")

# ---- Run ----
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
