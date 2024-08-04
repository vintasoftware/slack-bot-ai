import os
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slackstyler import SlackStyler

from project.use_cases.answer_slack_message_with_llm import AnswerSlackMessageUseCase
from project.llm.gpt import GPT_4o_MINI as LLM

app = App()
styler = SlackStyler()


app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

@app.event("app_mention")
def handle_app_mentions(body, say, logger):
    # health checking for now
    logger.info(body)
    say("What's up?")


@app.event("message")
def handle_message(message, say, logger):
    user, text = (message["user"], message["text"])
    _, *ai_messages = AnswerSlackMessageUseCase(llm_with_tools=LLM).execute(text)
    
    for message in ai_messages:
        say( styler.convert(message.content) ) if message.content else print(message)

slack_handler = SlackRequestHandler(app)