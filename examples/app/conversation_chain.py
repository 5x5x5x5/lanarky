from functools import lru_cache
from typing import Callable

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.templating import Jinja2Templates
from langchain import ConversationChain, OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel
import uvicorn

from lanarky.responses import StreamingResponse
#from lanarky.routing import LangchainRouter
from lanarky.testing import mount_gradio_app
from lanarky.websockets import WebsocketConnection


load_dotenv()




app = mount_gradio_app(FastAPI(title="ConversationChainDemo"))
templates = Jinja2Templates(directory="templates")




class QueryRequest(BaseModel):
   query: str




def conversation_chain_dependency() -> Callable[[], ConversationChain]:
   @lru_cache(maxsize=1)
   def dependency() -> ConversationChain:


       template = """The following is a friendly conversation between a human and an AI. The AI 
       is talkative and provides lots of specific details from its context. If the AI does not 
       know the answer to a question, it truthfully says it does not know.
    

       {history}


       Human: {input}
       AI:"""


       PROMPT = PromptTemplate(
           input_variables=["input", "history"],
           template=template
           )
  
       return ConversationChain(
           llm=OpenAI(
               temperature=0,
               streaming=True,
           ),
           verbose=True,
           prompt=PROMPT,
       )


   return dependency




conversation_chain = conversation_chain_dependency()




@app.post("/chat")
async def chat(
   request: QueryRequest,
   chain: ConversationChain = Depends(conversation_chain),
) -> StreamingResponse:
   return StreamingResponse.from_chain(
       chain, request.query, media_type="text/event-stream"
   )




@app.get("/")
async def get(request: Request):
   return templates.TemplateResponse("index.html", {"request": request})




@app.websocket("/ws")
async def websocket_endpoint(
   websocket: WebSocket, chain: ConversationChain = Depends(conversation_chain)
):
   connection = WebsocketConnection.from_chain(chain=chain, websocket=websocket)
   await connection.connect()




if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)




#def create_chain():
#    return ConversationChain(
#        llm=ChatOpenAI(
#            temperature=0,
#            streaming=True,
#        ),
#        verbose=True,
#    )


#app = mount_gradio_app(FastAPI(title="ConversationChainDemo"))
#templates = Jinja2Templates(directory="templates")
#chain = create_chain()


#@app.get("/")
#async def get(request: Request):
#    return templates.TemplateResponse("index.html", {"request": request})


#langchain_router = LangchainRouter(
#    langchain_url="/chat", langchain_object=chain, streaming_mode=1
#)
#langchain_router.add_langchain_api_route(
#    "/chat_json", langchain_object=chain, streaming_mode=2
#)
#langchain_router.add_langchain_api_websocket_route("/ws", langchain_object=chain)

#app.include_router(langchain_router)
