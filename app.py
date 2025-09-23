# app.py
from dotenv import load_dotenv
from openai import OpenAI
import json, os, requests, html as html_module, re
from pypdf import PdfReader
import gradio as gr
from fastapi import FastAPI, Request, Response
import uvicorn
from gradio.routes import mount_gradio_app
from datetime import datetime, timedelta

# Load env
load_dotenv(override=True)
app = FastAPI()
port = int(os.environ.get("PORT", 4000))

# ---------- Small notification helpers (optional) ----------
def push(text):
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": os.getenv("PUSHOVER_TOKEN"),
                "user": os.getenv("PUSHOVER_USER"),
                "message": text,
            },
            timeout=5
        )
    except Exception:
        pass

# ---------- Tools used by your assistant (same as you had) ----------
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

# ---------- Minimal Me class (keeps your existing assistant behavior) ----------
class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "CuraGuide"
        # try to load some local resources if present
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
            "If you don't know an answer, use the record_unknown_question tool. "
            "If the user provides an email, record it using record_user_details.\n\n"
            f"## Summary:\n{self.summary}\n\n## LinkedIn:\n{self.linkedin}\n"
        )

    def chat(self, message, history):
        # simple passthrough to OpenAI (keeps backward compatibility)
        flat_history = []
        for h in (history or []):
            if isinstance(h, list):
                flat_history.extend(h)
            else:
                flat_history.append(h)
        messages = [{"role":"system", "content": self.system_prompt()}] + flat_history + [
            {"role":"user", "content": message}
        ]
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

# ---------- Vaccination database ----------
# This is a comprehensive-but-not-exhaustive mapping. Adapt to your country's official schedule.
VACCINE_DB = {
    "bcg": {
        "disease": "Tuberculosis (BCG)",
        "standard": [{"age": "At birth", "dose": "Single dose (intradermal)"}],
        "notes": "Given at birth to prevent severe forms of TB in children."
    },
    "hepatitis b": {
        "disease": "Hepatitis B",
        "standard": [
            {"age": "At birth", "dose": "Dose 1"},
            {"age": "6 weeks", "dose": "Dose 2"},
            {"age": "14 weeks", "dose": "Dose 3"}
        ],
        "notes": "3-dose series; catch-up dosing available."
    },
    "dpt": {
        "disease": "Diphtheria, Pertussis (Whooping cough), Tetanus (DPT/DTaP/Tdap)",
        "standard": [
            {"age": "6 weeks", "dose": "Dose 1"},
            {"age": "10 weeks", "dose": "Dose 2"},
            {"age": "14 weeks", "dose": "Dose 3"},
            {"age": "15-18 months", "dose": "Booster"},
            {"age": "4-6 years", "dose": "Booster"}
        ],
        "notes": "Tdap booster in adolescence; adult booster every 10 years for tetanus."
    },
    "polio": {
        "disease": "Polio (OPV/IPV)",
        "standard": [
            {"age": "At birth (OPV0)", "dose": "OPV 0 (where applicable)"},
            {"age": "6 weeks", "dose": "Dose 1"},
            {"age": "10 weeks", "dose": "Dose 2"},
            {"age": "14 weeks", "dose": "Dose 3"},
            {"age": "4-6 years", "dose": "Booster"}
        ],
        "notes": "Follow national polio campaign guidance for supplementary immunization."
    },
    "mmr": {
        "disease": "Measles, Mumps, Rubella (MMR)",
        "standard": [
            {"age": "9-12 months", "dose": "Dose 1 (may vary by country)"},
            {"age": "15-18 months or 16-24 months", "dose": "Dose 2 / second dose"}
        ],
        "notes": "Important to achieve herd immunity; outbreak response often targets 6 months+ or up to older children."
    },
    "haemophilus influenzae type b": {
        "disease": "H. influenzae type b (Hib)",
        "standard": [
            {"age": "6 weeks", "dose": "Dose 1"},
            {"age": "10 weeks", "dose": "Dose 2"},
            {"age": "14 weeks", "dose": "Dose 3"}
        ],
        "notes": "Prevents meningitis and severe infections in infants."
    },
    "pneumococcal": {
        "disease": "Pneumococcal disease (PCV)",
        "standard": [
            {"age": "6 weeks", "dose": "Dose 1"},
            {"age": "10 weeks", "dose": "Dose 2"},
            {"age": "14 weeks", "dose": "Dose 3"}
        ],
        "notes": "Also recommended for older adults or high-risk individuals."
    },
    "rotavirus": {
        "disease": "Rotavirus",
        "standard": [
            {"age": "6 weeks", "dose": "Dose 1"},
            {"age": "10 weeks", "dose": "Dose 2"},
            {"age": "14 weeks", "dose": "Dose 3 (depending on vaccine)"}
        ],
        "notes": "Prevents severe diarrhoea in infants."
    },
    "influenza": {
        "disease": "Seasonal influenza",
        "standard": [
            {"age": "Annually", "dose": "Once yearly for everyone (esp. high risk)"}
        ],
        "notes": "Annual vaccination is recommended for high-risk groups and recommended more broadly by some authorities."
    },
    "hpv": {
        "disease": "Human Papillomavirus (HPV)",
        "standard": [
            {"age": "9-14 years", "dose": "2-dose schedule (depending on vaccine)"},
            {"age": "15+ years", "dose": "3-dose schedule (catch up)"}
        ],
        "notes": "Targets prevention of cervical and other cancers; usually recommended for adolescents."
    },
    "hepatitis a": {
        "disease": "Hepatitis A",
        "standard": [
            {"age": "Childhood/adult depending on program", "dose": "2 doses separated by 6-18 months"}
        ],
        "notes": "Used where Hep A risk is high or for travellers."
    },
    "cholera": {
        "disease": "Cholera",
        "standard": [
            {"age": "When indicated", "dose": "2 doses depending on product and schedule"}
        ],
        "notes": "Used for outbreak response or high-risk populations."
    },
    "covid": {
        "disease": "COVID-19",
        "standard": [
            {"age": "Per local guidance", "dose": "Primary series + boosters as recommended"}
        ],
        "notes": "Follow latest national guidance for boosters."
    },
    "typhoid": {
        "disease": "Typhoid",
        "standard": [
            {"age": "Where recommended", "dose": "Single dose (or multiple depending on product)"}
        ],
        "notes": "Used in some national programs or for travellers."
    },
    # add more vaccines as required...
}

# Map simple disease keywords -> vaccine keys
DISEASE_TO_VACCINE_MAP = {
    "measles": "mmr",
    "mumps": "mmr",
    "rubella": "mmr",
    "tuberculosis": "bcg",
    "tb": "bcg",
    "hepatitis b": "hepatitis b",
    "hep b": "hepatitis b",
    "diphtheria": "dpt",
    "pertussis": "dpt",
    "whooping cough": "dpt",
    "tetanus": "dpt",
    "polio": "polio",
    "rotavirus": "rotavirus",
    "pneumonia": "pneumococcal",
    "pneumococcal": "pneumococcal",
    "hib": "haemophilus influenzae type b",
    "influenza": "influenza",
    "flu": "influenza",
    "hpv": "hpv",
    "human papillomavirus": "hpv",
    "hepatitis a": "hepatitis a",
    "cholera": "cholera",
    "covid": "covid",
    "covid-19": "covid",
    "sars-cov-2": "covid",
    "typhoid": "typhoid",
    # vector-borne
    "dengue": None,
    "malaria": None,
    "zika": None,
    "chikungunya": None,
}

# ---------- Serper (or other) search for latest outbreaks ----------
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def fetch_serper_updates(search_query=None):
    # General fallback query
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

# ---------- Outbreak detection and schedule generation ----------
def extract_diseases_from_text(text):
    # Normalize and search for keywords (simple heuristics)
    text_lower = re.sub(r'[^a-z0-9 -]', ' ', html_module.unescape(text).lower())
    found = set()
    for keyword in DISEASE_TO_VACCINE_MAP.keys():
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            found.add(keyword)
    # dedupe and return
    return list(found)

def create_vaccination_schedule_from_updates(html_text):
    diseases = extract_diseases_from_text(html_text)
    # Standard schedule (summary of all vaccines)
    standard_parts = ["<b>Standard (basic) vaccination schedule — high level:</b><br><br>"]
    for key, v in VACCINE_DB.items():
        standard_parts.append(f"<b>{v['disease']}</b><br>")
        for entry in v["standard"]:
            standard_parts.append(f"- {entry['age']}: {entry['dose']}<br>")
        if v.get("notes"):
            standard_parts.append(f"<i>{v['notes']}</i><br>")
        standard_parts.append("<br>")
    standard_html = "".join(standard_parts)

    # Outbreak adjustments
    if not diseases:
        outbreak_html = "<i>No specific vaccine-preventable disease mentioned in the fetched updates.</i>"
    else:
        outbreak_parts = ["<b>Outbreak-detected adjustments & recommended drives:</b><br><br>"]
        now = datetime.utcnow()
        for d in diseases:
            vaccine_key = DISEASE_TO_VACCINE_MAP.get(d)
            pretty_name = d.title()
            outbreak_parts.append(f"<b>{pretty_name}</b><br>")
            if vaccine_key and vaccine_key in VACCINE_DB:
                # If vaccine exists, recommend urgent drive
                v = VACCINE_DB[vaccine_key]
                # Estimate schedule: immediate awareness, mass drive within 2 weeks, mop-up after 4-6 weeks
                awareness_date = now + timedelta(days=1)  # immediate
                mass_drive_start = now + timedelta(days=7)  # within 1 week
                mass_drive_end = now + timedelta(days=21)  # 3 weeks window
                mop_up_start = now + timedelta(days=28)
                mop_up_end = now + timedelta(days=42)
                target = "Target: infants/children and unvaccinated individuals; expand to adolescents/adults depending on local vulnerability."
                # More precise targeting for certain diseases
                if d in ("measles", "mumps", "rubella"):
                    target = "Target: children 6 months - 15 years (prioritise 9 months - 5 years)."
                if d in ("polio",):
                    target = "Target: all children under 5 years; follow national polio campaign guidance."
                outbreak_parts.append(f"Suggested vaccine: <b>{v['disease']}</b><br>")
                outbreak_parts.append(f"{target}<br>")
                outbreak_parts.append(f"Awareness campaign: start {awareness_date.strftime('%Y-%m-%d')} (immediately).<br>")
                outbreak_parts.append(f"Mass vaccination drive window: {mass_drive_start.strftime('%Y-%m-%d')} to {mass_drive_end.strftime('%Y-%m-%d')}.<br>")
                outbreak_parts.append(f"Mop-up / follow-up: {mop_up_start.strftime('%Y-%m-%d')} to {mop_up_end.strftime('%Y-%m-%d')}.<br>")
                outbreak_parts.append("<i>Note: follow vaccine eligibility rules for age/contraindications.</i><br><br>")
            else:
                # No vaccine (e.g., dengue/malaria) or vaccine not in DB => prevention and vector control
                if d in ("dengue", "malaria", "zika", "chikungunya"):
                    outbreak_parts.append("No universal vaccination program in many regions. Recommend vector control & prevention drives:<br>")
                    outbreak_parts.append("- Community awareness (remove standing water, nets, repellents).<br>")
                    outbreak_parts.append("- Rapid case detection & clinical care training within 48-72 hours.<br>")
                    outbreak_parts.append("- Consider targeted vaccination campaigns if an approved vaccine is recommended locally (check local authority).<br><br>")
                else:
                    outbreak_parts.append("No specific vaccine mapping in local DB; recommend contacting local public health authority for appropriate interventions.<br><br>")
        outbreak_html = "".join(outbreak_parts)

    combined = "<div style='max-height:420px;overflow:auto;'>" + standard_html + "<hr>" + outbreak_html + "</div>"
    return combined

# ---------- Styling (teal / white contrast) ----------
custom_css = r"""
body {
    background: linear-gradient(135deg, #0ea5a4 0%, #06b6d4 100%) !important;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    color: #042f2e;
}
#profile-sidebar {
    background: #044e4b;
    color: #fff;
    min-width: 260px;
    max-width: 320px;
    padding: 24px;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(2,51,50,0.25);
    margin-right: 16px;
}
#chat-display {
    background: #f8fffd;
    border-radius: 12px;
    padding: 12px;
    height: 420px;
    overflow: hidden;
}
.chat-window { height: 100%; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; padding-right: 8px; }
.bubble { max-width: 85%; padding: 12px 16px; border-radius: 14px; line-height: 1.45; box-shadow: 0 3px 11px rgba(2,51,50,0.06); }
.bubble.user { align-self: flex-end; background: #10b981 !important; color: #fff !important; }
.bubble.assistant { align-self: flex-start; background: #a7f3d0 !important; color: #064e3b !important; }
.bubble.system { align-self: center; background: #e6fffa; color: #065f46; }
#updates-display, #schedule-display { background: #fff; border-radius: 10px; padding: 12px; height: 200px; overflow-y: auto; box-shadow: 0 6px 24px rgba(2,51,50,0.08); color: #044e4b; font-size: 14px; line-height: 1.4; }
#message-input textarea { border-radius: 10px !important; border: 2.5px solid #06b6d4 !important; background: #fff !important; color: #064e3b !important; height: 48px !important; }
#send-btn { background: linear-gradient(90deg,#10b981 40%,#06b6d4 100%) !important; color: #fff !important; border-radius: 10px !important; font-weight: 600; border: none; }
.small-timestamp { color: #065f46; font-size: 12px; font-style: italic; }
"""

# ---------- Gradio UI ----------
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    with gr.Row():
        with gr.Column(elem_id="profile-sidebar", scale=0.20):
            # Replace or set your logo image path
            if os.path.exists("image.jpg"):
                gr.Image("image.jpg", elem_id="logo", show_label=False, width=120)
            else:
                gr.Markdown("### CuraGuide\nAI-Powered Public Health Assistant")
            gr.Markdown("<div class='tagline'>Reliable disease awareness and guidance</div>")
            gr.Markdown("---")
            btn_updates = gr.Button("Refresh Latest Updates")
            btn_schedule = gr.Button("Create Vaccination Schedule")
            gr.Markdown("<div style='font-size:13px;color:#d1fae5'>Status: live</div>")

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
    # This is a minimal Twilio-compatible webhook response.
    # If using Twilio, configure 'Messaging' webhook to point here.
    form = await request.form()
    user_text = form.get("Body", "")
    reply = me.chat(user_text, [])
    # Build TwiML
    twiml = f"<?xml version='1.0' encoding='UTF-8'?><Response><Message>{html_module.escape(reply)}</Message></Response>"
    return Response(content=twiml, media_type="application/xml")

# ---------- Run server ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)