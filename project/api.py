import os
from fastapi import FastAPI, Query
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import uvicorn

from dotenv import load_dotenv
load_dotenv()

from .confluence import DataLoader
from .vector_store import VectorStore

app = FastAPI()

LLM = ChatOpenAI(model="gpt-4o-mini")

@app.post("/confluence/{space_key}")
def ingest_knowledge_base(space_key: str):
    data_loader, db = (DataLoader(), VectorStore())
    
    docs = data_loader.load(space_key=space_key)
    processed_docs = data_loader.split_docs(docs)
    
    db.create_db(processed_docs)
    
    return {"message": "ok!"}

@app.get("/confluence/")
def query_knowledge_base(query: str = Query(...)):
    db = VectorStore()
    results = db.query_db(query)
    retriever = db.get_store().as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(f"{doc.page_content}\nsource:{doc.metadata.get('source')}" for doc in docs)

    from .prompting import QNA_PROMPT, ConversationalRetrievalChain
    qa = ConversationalRetrievalChain.from_llm(
        llm=LLM, 
        retriever=retriever, 
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QNA_PROMPT}
    )
   
    input = {"question": query, "chat_history": []}
    retrieved = qa.invoke(input=input)
    sources = list(set(doc.metadata.get("source") for doc in retrieved["source_documents"]))
    answer = retrieved["answer"]

    refine_prompt_messages = [
        (
            "system",
            "Based on the context provided, refine the answer to the user's question, which is about Vinta's internal documentation. Include source links whenever possible.",
        ),
        ("human", f"context: {format_docs(retrieved['source_documents'])} \n\n answer to user question:{answer}"),
   ]

    refined = LLM.invoke(refine_prompt_messages)

    return {"answer": answer, "refined_answer": refined.content, "links": sources}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7999)