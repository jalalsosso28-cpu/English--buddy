import json, re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = anthropic.Anthropic()  # lit ANTHROPIC_API_KEY dans l'environnement
MODEL = "claude-sonnet-5-5"

LEVELS = {
    "A1": "A1 (very basic): very short sentences, only the most common words, speak slowly.",
    "A2": "A2: simple sentences, everyday vocabulary, present and simple past.",
    "B1": "B1: clear standard English, common tenses, some less common words.",
    "B2": "B2: natural English, varied grammar, idioms used sparingly.",
    "C1": "C1: fluent, nuanced English with phrasal verbs and idioms.",
    "C2": "C2: native-like English, subtle expressions, humour, cultural references.",
}

SYSTEM = """You are a friendly English conversation partner for a French-speaking learner.
Topic: {topic}. Level: {level}
Rules:
- Match the level strictly. Reply in 2-3 short sentences, then ask ONE follow-up question.
- If the learner made a mistake, start with a line: "Correction: <corrected sentence>" and a very short tip in French. If there is no mistake, skip it.
- Stay in the topic (role-play the situation: waiter, receptionist, ticket seller, etc.).
- If the learner writes in French, give the English version and ask them to repeat it."""

class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    topic: str = "everyday conversation"
    level: str = "A2"
    messages: list[Msg]

class QuizReq(BaseModel):
    topic: str = "everyday conversation"
    level: str = "A2"
    n: int = 5

@app.post("/chat")
def chat(req: ChatReq):
    r = client.messages.create(
        model=MODEL, max_tokens=400,
        system=SYSTEM.format(topic=req.topic, level=LEVELS.get(req.level, LEVELS["A2"])),
        messages=[m.model_dump() for m in req.messages],
    )
    return {"reply": r.content[0].text}

@app.post("/quiz")
def quiz(req: QuizReq):
    prompt = (f"Create {req.n} multiple-choice vocabulary questions for a French speaker learning English. "
              f"Topic: {req.topic}. CEFR level: {req.level}. Each question asks the meaning of an English word or "
              "expression in context. Reply with ONLY a JSON array, no markdown. Each item: "
              '{"q": "English sentence with the target word in **bold**", "options": [4 short French translations], '
              '"answer": index 0-3, "explain": "one short sentence in French"}')
    r = client.messages.create(model=MODEL, max_tokens=1200, messages=[{"role": "user", "content": prompt}])
    text = re.sub(r"```json|```", "", r.content[0].text).strip()
    return {"questions": json.loads(text)}
