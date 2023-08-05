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


WELCOME_MSG = """
    Chatbot is ready. You can ask it: \n
    - Questions relevant to the knowledge doc\n
    - (Bonus Feature) ask it to check how relevant the previous answer was to the knowledge doc ie if the information/knowledge from the knowledge doc actually supports the chatbot's answer\n
    To use this feature, simply type 'Evaluate' in the bot after you have asked it the question. Type 'Evaluate verbose' to see the underlying LLM prompt as well\n
    The bot will answer YES/NO depending on whether the retrieved info from the knowledge doc supports the bot's answer or not\n
    Unpaid openAI keys are rate limited and hence would slow down the chatbot\n
    """

EVAL_DESC = "#For verbosity, the entire prompt sent to the LLM for answer checking is shown below\n"

ANS_EVAL_TEMPLATE = """
Please tell if a given piece of information is supported by the context.\n
You need to answer with either YES or NO.\n
Answer YES if any of the context supports the information, even if most of the context is unrelated. Some examples are provided below. \n\n
Information: Apple pie is generally double-crusted.\n
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples. \n
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.\n
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).\n
Answer: YES\n
Information: Apple pies tastes bad.\n
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples. \n
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.\n
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).\n
Answer: NO\n
Information: {query_str}\n
Context: {context_str}\n
Answer: 
"""

ANS_EVAL_PROMPT = PromptTemplate(template=ANS_EVAL_TEMPLATE, input_variables=["query_str","context_str"])

@cl.on_chat_start
async def init():

    loader = UnstructuredFileLoader("knowledgedoc.txt")

    msg = cl.Message(content=f"Processing file...")
    await msg.send()

    unsplitdoc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True)
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
    prompt = ANS_EVAL_PROMPT
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
        eval_ans = await eval_chain.arun({'query_str':eval_ans,'context_str':eval_context})
        await cl.Message(content="CORRECT" if eval_ans=="YES" else "INCORRECT").send()
    
    if message == "Evaluate verbose":
        eval_chain = cl.user_session.get("evalchain")
        eval_context = cl.user_session.get("prev_context")
        eval_ans = cl.user_session.get("prevans")
        await cl.Message(content=EVAL_DESC).send()
        await cl.Message(content=ANS_EVAL_PROMPT.format(query_str = eval_ans, context_str = eval_context.page_content)).send()
        eval_ans = await eval_chain.arun({'query_str':eval_ans,'context_str':eval_context})
        await cl.Message(content=eval_ans).send()

    else:
        qa = cl.user_session.get("qachain") 
        context_finder= cl.user_session.get("retriever")

        context = context_finder.get_relevant_documents(message)
        full_context = Document(page_content="",metadata={})
        for entry in context:
            full_context.page_content = full_context.page_content+ entry.page_content
        cl.user_session.set("prev_context",full_context)
        cl.user_session.set("prev_ques",message)
        res = await qa.arun({"query": message},callbacks=[cl.AsyncLangchainCallbackHandler()])
        await cl.Message(content=res).send()
        cl.user_session.set("prevans",res)




