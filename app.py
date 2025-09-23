from dotenv import load_dotenv
from openai import OpenAI
import json, os, requests, html as html_module
from pypdf import PdfReader
import gradio as gr
from fastapi import FastAPI, Request
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

# ---------- Custom CSS (updated color scheme: teal/green contrast with white) ----------
custom_css = r"""
body {
    background: linear-gradient(135deg, #0ea5a4 0%, #06b6d4 100%) !important; /* teal -> cyan */
    font-family: 'Segoe UI', Tahoma, sans-serif;
    color: #042f2e;
}
#profile-sidebar {
    background: #044e4b;
    color: #fff;
    min-width: 260px;
    max-width: 320px;
    padding: 24px;
    border-radius: 17px;
    box-shadow: 0 8px 32px rgba(2,51,50,0.25);
    margin-right: 26px;
}
#profile-sidebar h3 { color: #a7f3d0; margin-bottom: 6px; }
#profile-sidebar .tagline { color: #d1fae5; font-size: 13px; }
.placeholder-avatar {
    width: 82px; height: 82px;
    border-radius: 50%;
    background: #34d399;
    display: flex; align-items: center; justify-content: center;
    color: #064e3b; font-weight: 700; font-size: 28px;
    margin-bottom: 16px;
    box-shadow: 0 0 0 4px #a7f3d050;
}
#logo { margin-bottom: 20px; }
#chat-display { 
    background: #f8fffd;  /* very light - contrasts with teal */
    border-radius: 14px; 
    padding: 14px; 
    box-shadow: 0 6px 24px rgba(2,51,50,0.06);
    height: 420px;
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
    box-shadow: 0 3px 11px rgba(2,51,50,0.06);
}
.bubble.user {
    align-self: flex-end;
    background: #10b981 !important;  /* green */
    color: #fff !important;
}
.bubble.assistant {
    align-self: flex-start;
    background: #a7f3d0 !important;  /* pale mint */
    color: #064e3b !important;
}
.bubble.system {
    align-self: center;
    background: #e6fffa;
    color: #065f46;
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
    border: 2.5px solid #06b6d4 !important;
    resize: none !important;
    height: 48px !important;
    overflow-y: auto !important;
    background: #ffffff !important;
    color: #064e3b !important;
}
#send-btn {
    background: linear-gradient(90deg,#10b981 40%,#06b6d4 100%) !important;
    color: #fff !important;
    border-radius: 12px !important;
    font-weight: 600;
    letter-spacing: 0.04em;
    border: none;
}
#updates-display {
    background: #ffffff;
    border-radius: 14px;
    padding: 16px;
    height: 200px;
    overflow-y: auto;
    box-shadow: 0 6px 24px rgba(2,51,50,0.08);
    font-size: 14px;
    line-height: 1.4;
    color: #044e4b;
}
#schedule-display {
    background: #ffffff;
    border-radius: 14px;
    padding: 16px;
    height: 200px;
    overflow-y: auto;
    box-shadow: 0 6px 24px rgba(2,51,50,0.08);
    font-size: 14px;
    line-height: 1.4;
    color: #044e4b;
}
#updates-display a { color: #064e3b; text-decoration: none; }
#updates-display a:hover { text-decoration: underline; }
.small-timestamp { color: #065f46; font-size: 12px; font-style: italic; }
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
    try:
        response = requests.post("https://google.serper.dev/search", headers=headers, json=json_data, timeout=10)
    except Exception:
        return "<b>Could not fetch live updates at this time.</b>"
    if response.status_code == 200:
        data = response.json()
        snippets = []
        for item in data.get("organic", [])[:6]:  # top 6 results
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            date = item.get("date") or item.get("published") or ""
            date_html = f"<br><span class='small-timestamp'>Updated: {date}</span>" if date else ""
            snippets.append(f"<b>{title}</b><br>{snippet}{date_html}<br><a href='{link}' target='_blank'>{link}</a><br><br>")
        return "".join(snippets)
    else:
        return "<b>Could not fetch live updates at this time.</b>"

# ---------- Create Vaccination Schedule from Updates (simple heuristic) ----------
COMMON_VACCINATION_GUIDE = {
    "measles": "MMR vaccine: 1st dose at 9-12 months, 2nd dose at 15-18 months or at school entry.",
    "diphtheria": "DTaP vaccine: multiple doses in infancy (2,4,6 months) plus boosters at 15-18 months and 4-6 years.",
    "tetanus": "DTaP/Tdap per childhood schedule; adult boosters every 10 years.",
    "pertussis": "Part of DTaP/Tdap schedule; boosters for adolescents and pregnant women each pregnancy.",
    "influenza": "Seasonal flu vaccine: once yearly for most age groups; recommended for high-risk individuals.",
    "hepatitis a": "HepA vaccine: 2 doses separated by 6-18 months depending on schedule.",
    "hepatitis b": "HepB vaccine: typically a 3-dose series in infancy or catch-up dosing for adults.",
    "covid": "COVID-19 vaccine: follow local health authority guidance for primary series and boosters.",
    "cholera": "Oral cholera vaccine: given as 2 doses depending on vaccine brand; follow local guidance.",
    "polio": "OPV/IPV schedule in infancy with booster doses as per national program.",
}


def extract_diseases_from_text(html_text):
    # Simple matching against keywords in COMMON_VACCINATION_GUIDE
    found = set()
    lowered = html_module.unescape(html_text).lower()
    for disease in COMMON_VACCINATION_GUIDE.keys():
        if disease in lowered:
            found.add(disease)
    # Heuristic: also look for a few common disease names not in the map
    extras = ["malaria", "dengue", "zika", "chikungunya"]
    for ex in extras:
        if ex in lowered:
            found.add(ex)
    return list(found)


def create_vaccination_schedule_from_updates(html_text):
    diseases = extract_diseases_from_text(html_text)
    if not diseases:
        return "<i>No clear vaccine-preventable disease mentions found in the fetched updates. If you have a specific disease in mind, ask the bot.</i>"
    parts = ["<b>Suggested vaccination/precaution schedule (high-level):</b><br><br>"]
    for d in diseases:
        guide = COMMON_VACCINATION_GUIDE.get(d)
        if guide:
            parts.append(f"<b>{d.title()}</b>: {guide}<br><br>")
        else:
            # For vector-borne diseases give prevention advice
            parts.append(f"<b>{d.title()}</b>: No universal vaccine in many countries — recommend prevention measures (mosquito avoidance, nets, repellents), and consult local health authority.<br><br>")
    parts.append("<i>Note: This is a general guide. Always follow local public health authority recommendations and consult a healthcare professional for personalised schedules.</i>")
    return "".join(parts)

# ---------- Gradio UI ----------
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row():
        with gr.Column(elem_id="profile-sidebar", scale=0.20):
            gr.Image("image.jpg", elem_id="logo", show_label=False, width=140)
            gr.Markdown("### CuraGuide\nAI-Powered Public Health Assistant")
            gr.Markdown("<div class='tagline'>Reliable disease awareness and guidance</div>")
            gr.Markdown("---")
            gr.Markdown("**Quick actions**")
            btn_updates = gr.Button("Refresh Latest Updates")
            btn_schedule = gr.Button("Create Vaccination Schedule")
            gr.Markdown("<div style='font-size:13px;color:#d1fae5'>Status: live</div>")

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
            gr.Markdown("### Suggested Vaccination Schedule")
            schedule_html = gr.HTML("<i>Press 'Create Vaccination Schedule' to generate from updates</i>", elem_id="schedule-display")

            def load_updates():
                return fetch_serper_updates()

            def generate_schedule(_):
                updates = fetch_serper_updates()
                schedule = create_vaccination_schedule_from_updates(updates)
                return updates, schedule

            demo.load(load_updates, outputs=updates_html)
            btn_updates.click(lambda: fetch_serper_updates(), outputs=updates_html)
            btn_schedule.click(generate_schedule, inputs=[], outputs=[updates_html, schedule_html])

# Serve Gradio app at "/"
app = mount_gradio_app(app, demo, path="/")

# ---------- Optional Twilio webhook example (if you choose low-code path) ----------
@app.post("/twilio-whatsapp")
async def twilio_whatsapp(request: Request):
    form = await request.form()
    user_text = form.get("Body")
    reply = me.chat(user_text, [])
    # Return simple TwiML XML response
    twiml = f"<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Message>{html_module.escape(reply)}</Message></Response>"
    return gr.responses.HTML(twiml)

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
