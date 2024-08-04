import uvicorn

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Query, Request

from project.slack import slack_handler
from project.llm.gpt import GPT_4o_MINI as LLM

from project.use_cases.query_confluence import QueryConfluenceUseCase
from project.use_cases.ingest_confluence_space import IngestConfluenceSpaceUseCase

app = FastAPI()


@app.post("/confluence/{space_key}")
def ingest_knowledge_base(space_key: str):
    return IngestConfluenceSpaceUseCase(space_key=space_key).execute()


@app.get("/confluence/")
def query_knowledge_base(query: str = Query(...)):
    return QueryConfluenceUseCase(llm=LLM).execute(query)


@app.post("/slack/events/")
async def slack_webhook(req: Request):
    # sets up url verification for the slack app, probably broken
    # need to fix when running ngrok again
    challenge = req.get("challenge")

    if challenge:
        return {"challenge": challenge}

    return await slack_handler.handle(req)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7999)
