from fastapi import FastAPI, Request

# Templating and Static Files.
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# POST validations.
from pydantic import BaseModel

# NLU.
from backend.nlu import intent_classifier, action_predictor

app = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
def homepage(request: Request):
    return templates.TemplateResponse("homepage.html", context={
        "request": request
    })


class Chat(BaseModel):
    """Validating POST"""
    user_input: str


SESSION_INTENTS = []


@app.post("/chat")
def chat_with_bot(chat: Chat):
    """
    Generates Bot Responses.
    """
    # POST Data.
    user_input = chat.user_input

    # Intents Detected.
    intent_detected = intent_classifier(user_input)
    SESSION_INTENTS.append(intent_detected)

    # Next Action Detection.
    next_action = action_predictor(SESSION_INTENTS)

    # Empty Intent List when session ends.
    if next_action == "end":
        SESSION_INTENTS.clear()

    # BOT Response.
    responses = {
        "city": ["which city are you going to?", "which city are leaving from?"],
        "travel_date": "what is the travel date?",
        "travel_duration": "what is the travel duration?",
        "end": "thanks, the details will be shared soon."
    }

    # City Response Logic.
    if intent_detected != "city" and next_action == "city":
        # City to
        response = responses[next_action][0]
    elif intent_detected == "city" and next_action == "city":
        # City from
        response = responses[next_action][1]
    else:
        response = responses[next_action]

    return {
        "bot_response": response
    }
