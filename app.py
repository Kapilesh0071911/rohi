# ui_vaccination_schedule_app.py

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
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
PORT = int(os.getenv("PORT", 4000))

app = FastAPI()

# ---------- Vaccine schedule for India (UIP) ----------
# Mandatory/basic vaccines as per Indian National Immunization Schedule

VACCINE_DB_INDIA = {
    "bcg": {
        "disease": "Tuberculosis (severe forms in children)",
        "standard": [
            {"age": "At birth", "dose": "BCG — single dose"}
        ],
        "notes": "Given as soon as possible after birth."
    },
    "hepatitis b (birth dose)": {
        "disease": "Hepatitis B",
        "standard": [
            {"age": "Within 24 hours of birth", "dose": "Hepatitis B – birth dose"}
        ],
        "notes": "Part of complete 3-dose series."
    },
    "opv-0": {
        "disease": "Polio",
        "standard": [
            {"age": "At birth / within first 15 days", "dose": "OPV 0 (zero dose)"}
        ],
        "notes": "Supplemented by later OPV/IPV doses."
    },
    "pentavalent (DPT + HepB + Hib)": {
        "disease": "Diphtheria, Pertussis, Tetanus, Hepatitis B, Hib",
        "standard": [
            {"age": "6 weeks", "dose": "1st dose"},
            {"age": "10 weeks", "dose": "2nd dose"},
            {"age": "14 weeks", "dose": "3rd dose"}
        ],
        "notes": "Three doses in infancy as per UIP."
    },
    "ipv / fractional ipv": {
        "disease": "Polio (IPV / fIPV)",
        "standard": [
            {"age": "6 weeks", "dose": "IPV / fractional IPV – dose 1"},
            {"age": "14 weeks", "dose": "IPV / fractional IPV – dose 2"}
        ],
        "notes": "Given along with OPV per UIP."
    },
    "rotavirus vaccine": {
        "disease": "Rotavirus",
        "standard": [
            {"age": "6 weeks", "dose": "1st dose"},
            {"age": "10 weeks", "dose": "2nd dose"},
            {"age": "14 weeks", "dose": "3rd dose (if vaccine formulation requires)"}
        ],
        "notes": "Prevents severe diarrhoea caused by rotavirus."
    },
    "pneumococcal conjugate vaccine (pcv)": {
        "disease": "Pneumococcal disease",
        "standard": [
            {"age": "6 weeks", "dose": "1st dose"},
            {"age": "14 weeks", "dose": "2nd dose"}
            # Some states may use 3rd primary or booster
        ],
        "notes": "Protects against pneumonia, meningitis."
    },
    "measles-rubella (mr)": {
        "disease": "Measles and Rubella",
        "standard": [
            {"age": "9-12 months", "dose": "1st dose MR"},
            {"age": "16-24 months", "dose": "2nd dose MR / MR Booster"}
        ],
        "notes": "Ensures immunity in early childhood."
    },
    "je vaccine": {
        "disease": "Japanese Encephalitis (in endemic districts)",
        "standard": [
            {"age": "9-12 months", "dose": "JE – 1st dose"},
            {"age": "16-24 months", "dose": "JE – 2nd dose (if part of state schedule)"}
        ],
        "notes": "Only in JE endemic areas."
    },
    "dtp / dpt booster": {
        "disease": "Diphtheria, Pertussis, Tetanus (Boosters)",
        "standard": [
            {"age": "16-24 months", "dose": "1st Booster DPT"},
            {"age": "5-6 years", "dose": "2nd Booster DPT"}
        ],
        "notes": "Boost immunity beyond infancy."
    },
    "polio booster": {
        "disease": "Polio",
        "standard": [
            {"age": "16-24 months", "dose": "Polio booster / OPV / IPV"}
        ],
        "notes": "Also given via national polio campaigns."
    }
    # You can add more (HPV, influenza etc.) if your region includes them
}

# Keyword → vaccine key mapping (for outbreak detection)
DISEASE_TO_VACCINE_INDIA = {
    "measles": "measles-rubella (mr)",
    "rubella": "measles-rubella (mr)",
    "polio": "polio booster",
    "tuberculosis": "bcg",
    "tb": "bcg",
    "hepatitis b": "hepatitis b (birth dose)",
    "diphtheria": "pentavalent (DPT + HepB + Hib)",
    "pertussis": "pentavalent (DPT + HepB + Hib)",
    "whooping cough": "pentavalent (DPT + HepB + Hib)",
    "tetanus": "pentavalent (DPT + HepB + Hib)",
    "hib": "pentavalent (DPT + HepB + Hib)",
    "rotavirus": "rotavirus vaccine",
    "pneumococcal": "pneumococcal conjugate vaccine (pcv)",
    "japanese encephalitis": "je vaccine"
    # Add more if needed
}

# ---------- Assistant and outbreak fetching (same structure as earlier) ----------

class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "CuraGuide"

    def system_prompt(self):
        return f"You are {self.name}, an AI-powered public health assistant. Provide disease awareness and vaccination advice."

    def chat(self, message, history):
        flat_history = []
        for h in (history or []):
            if isinstance(h, list):
                flat_history.extend(h)
            else:
                flat_history.append(h)
        messages = [{"role":"system", "content": self.system_prompt()}] + flat_history + [{"role":"user","content":message}]
        resp = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        return resp.choices[0].message.content

me = Me()

def fetch_updates_india():
    if not SERPER_API_KEY:
        return "<b>API key missing. Cannot fetch outbreak updates.</b>"
    try:
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        body = {"q": "recent disease outbreaks India 2025", "gl":"in", "hl":"en"}
        r = requests.post("https://google.serper.dev/search", headers=headers, json=body, timeout=12)
    except Exception as e:
        return f"<b>Error fetching updates: {html_module.escape(str(e))}</b>"
    if r.status_code != 200:
        return f"<b>Could not fetch updates (status {r.status_code})</b>"
    data = r.json()
    snippets = []
    for item in data.get("organic", [])[:8]:
        t = item.get("title","")
        s = item.get("snippet","")
        link = item.get("link","")
        date = item.get("date") or item.get("published") or ""
        date_html = f"<br><span class='small-timestamp'>Updated: {date}</span>" if date else ""
        snippets.append(f"<b>{html_module.escape(t)}</b><br>{html_module.escape(s)}{date_html}<br><a href='{link}' target='_blank'>{link}</a><br><br>")
    if not snippets:
        return "<b>No recent outbreak updates found.</b>"
    return "".join(snippets)

def extract_diseases(text):
    text_low = html_module.unescape(text).lower()
    found = set()
    for kw in DISEASE_TO_VACCINE_INDIA.keys():
        if re.search(r'\b' + re.escape(kw) + r'\b', text_low):
            found.add(kw)
    return list(found)

def create_schedule_india(html_updates):
    # Standard schedule
    standard = ["<b>India UIP Basic Vaccination Schedule (Mandatory / Basic Vaccines):</b><br><br>"]
    for key, v in VACCINE_DB_INDIA.items():
        standard.append(f"<b>{v['disease']}</b><br>")
        for dose in v["standard"]:
            standard.append(f"- {dose['age']}: {dose['dose']}<br>")
        if v.get("notes"):
            standard.append(f"<i>{v['notes']}</i><br>")
        standard.append("<br>")

    # Outbreak adjustments
    diseases = extract_diseases(html_updates)
    if not diseases:
        out_html = "<i>No vaccine-preventable disease outbreaks detected in updates.</i>"
    else:
        out_html_parts = ["<b>Outbreak-triggered vaccination drive suggestions:</b><br><br>"]
        now = datetime.utcnow()
        for d in diseases:
            vaccine_key = DISEASE_TO_VACCINE_INDIA.get(d)
            pretty = d.title()
            out_html_parts.append(f"<b>{pretty}</b><br>")
            if vaccine_key and vaccine_key in VACCINE_DB_INDIA:
                v = VACCINE_DB_INDIA[vaccine_key]
                # Suggest timeline
                awareness = now + timedelta(days=1)
                drive_start = now + timedelta(days=7)
                drive_end = now + timedelta(days=21)
                mop_up = now + timedelta(days=30)
                target = "All children according to schedule; catch-up for unvaccinated; consider older age groups if data shows susceptibility."
                # If specific disease, customize
                if d in ("measles", "rubella"):
                    target = "Children 6 months-15 years (prioritise 9-12 months) in affected geographic area."
                if d == "polio":
                    target = "Infants and children under 5 years; heighten OPV/IPV drive."

                out_html_parts.append(f"Vaccine: <b>{v['disease']}</b><br>")
                out_html_parts.append(f"Target group: {target}<br>")
                out_html_parts.append(f"Awareness campaign: {awareness.strftime('%Y-%m-%d')}<br>")
                out_html_parts.append(f"Vaccination drive (mass): {drive_start.strftime('%Y-%m-%d')} to {drive_end.strftime('%Y-%m-%d')}<br>")
                out_html_parts.append(f"Mop-up / follow-up: {mop_up.strftime('%Y-%m-%d')}<br>")
                out_html_parts.append("<i>Follow local age eligibility and vaccine availability.</i><br><br>")
            else:
                out_html_parts.append("No schedule mapping available; recommend local health authority guidance.<br><br>")
        out_html = "".join(out_html_parts)

    combined = "<div style='max-height:420px;overflow:auto;'>" + "".join(standard) + "<hr>" + out_html + "</div>"
    return combined

# ---------- Simple UI and app ----------

custom_css = r"""
body {
    background: linear-gradient(135deg, #0ea5a4 0%, #06b6d4 100%) !important;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    color: #042f2e;
}
#updates-display, #schedule-display {
    background: #fff;
    border-radius: 10px;
    padding: 12px;
    height: 220px;
    overflow-y: auto;
    box-shadow: 0 4px 16px rgba(2,51,50,0.1);
    color: #044e4b;
    font-size: 14px;
    line-height: 1.4;
}
.small-timestamp {
    color: #065f46;
    font-size: 12px;
    font-style: italic;
}
#send-btn { background: linear-gradient(90deg,#10b981 40%,#06b6d4 100%) !important; color: #fff !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("## CuraGuide – Vaccination Schedule & Outbreak Alerts")

    updates_html = gr.HTML("<i>Loading outbreak updates...</i>", elem_id="updates-display")
    schedule_html = gr.HTML("<i>Schedule will appear here</i>", elem_id="schedule-display")
    btn = gr.Button("Generate / Update Vaccination Schedule")

    def on_click(_):
        upd = fetch_updates_india()
        sched = create_schedule_india(upd)
        return upd, sched

    demo.load(lambda: fetch_updates_india(), outputs=updates_html)
    btn.click(on_click, inputs=None, outputs=[updates_html, schedule_html])

app = mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
