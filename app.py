import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings , OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import chainlit as cl

os.environ['COHERE_API_KEY']='Ppe0NkciJqYoxhKeoe85HIfu5lZU3Z29vrsmbzUZ'

@cl.on_chat_start
async def init():

    loader = UnstructuredFileLoader("knowledgedoc.txt")

    msg = cl.Message(content=f"Processing file...")
    await msg.send()

    unsplitdoc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True)
    splitdocs = text_splitter.split_documents(unsplitdoc)

    oaikey = await cl.AskUserMessage(content="What is your OpenAI API KEY?", timeout=100,raise_on_timeout=True).send()
    if oaikey:
        await cl.Message(
            content=f"Your key is: {oaikey['content']}",
        ).send()
    os.environ['OPENAI_API_KEY']=oaikey['content']

    emb_model=CohereEmbeddings()
    vectorstore = FAISS.from_documents(splitdocs, emb_model)
    retriever=vectorstore.as_retriever()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever, memory=memory,verbose=True)

    msg = cl.Message(content=f"Chatbot is ready.")
    await msg.send()
    cl.user_session.set("qachain", qa)

@cl.on_message
async def main(message: str):
    qa = cl.user_session.get("qachain") 
    res = await qa.arun({"question": message},callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res).send()


