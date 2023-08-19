import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings , OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA , LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.docstore.document import Document
import chainlit as cl
from prompt import WELCOME_MSG , EVAL_DESC, ANS_EVAL_PROMPT as evalprompt


@cl.on_chat_start
async def init():

    loader = UnstructuredFileLoader("knowledgedoc.txt")

    msg = cl.Message(content=f"Processing file...")
    await msg.send()

    unsplitdoc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30, add_start_index=True)
    splitdocs = text_splitter.split_documents(unsplitdoc)

    oaikey = await cl.AskUserMessage(content="What is your OpenAI API KEY?",timeout=300,raise_on_timeout=True).send()
    if oaikey:
        await cl.Message(
            content=f"Your key is: {oaikey['content']}",
        ).send()
    os.environ['OPENAI_API_KEY']=oaikey['content']

    emb_model=OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splitdocs, emb_model)
    retriever=vectorstore.as_retriever()

    qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=False,
    verbose=True
    )

    eval_chain = LLMChain(
    llm=OpenAI(),
    prompt = evalprompt
    )

    msg = cl.Message(content=WELCOME_MSG)
    await msg.send()
    cl.user_session.set("evalchain",eval_chain)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("qachain", qa)

    

@cl.on_message
async def main(message: str):

    if message == "Evaluate":
        eval_chain = cl.user_session.get("evalchain")
        eval_context = cl.user_session.get("prev_context")
        eval_ans = cl.user_session.get("prevans")
        eval_res = await eval_chain.arun({'query_str':eval_ans,'context_str':eval_context})
        await cl.Message(content="CORRECT" if eval_res=="YES" else "INCORRECT").send()
    
    elif message == "Evaluate verbose":
        eval_chain = cl.user_session.get("evalchain")
        eval_context = cl.user_session.get("prev_context")
        eval_ans = cl.user_session.get("prevans")
        await cl.Message(content=EVAL_DESC).send()
        await cl.Message(content=evalprompt.format(query_str = eval_ans, context_str = eval_context.page_content)).send()
        eval_res = await eval_chain.arun({'query_str':eval_ans,'context_str':eval_context})
        await cl.Message(content=eval_res).send()

    else:
        qa = cl.user_session.get("qachain") 
        context_finder= cl.user_session.get("retriever")

        context = context_finder.get_relevant_documents(message)
        full_context = Document(page_content="",metadata={})
        for entry in context:
            full_context.page_content = full_context.page_content+ entry.page_content
            await cl.Message(content="chunk det").send()
        cl.user_session.set("prev_context",full_context)
        cl.user_session.set("prev_ques",message)
        res = await qa.arun({"query": message},callbacks=[cl.AsyncLangchainCallbackHandler()])
        cl.user_session.set("prevans",res)
        await cl.Message(content=res).send()
        




