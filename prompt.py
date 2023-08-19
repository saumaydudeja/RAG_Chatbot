from langchain.prompts import PromptTemplate

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

WELCOME_MSG = """
    Chatbot is ready. You can ask it: \n
    - Questions relevant to the knowledge doc\n
    - (Bonus Feature) ask it to check how relevant the previous answer was to the knowledge doc ie if the information/knowledge from the knowledge doc actually supports the chatbot's answer\n
    To use this feature, simply type 'Evaluate' in the bot after you have asked it the question. Type 'Evaluate verbose' to see the underlying LLM prompt as well\n
    The bot will answer YES/NO depending on whether the retrieved info from the knowledge doc supports the bot's answer or not\n
    Unpaid openAI keys are rate limited and hence would slow down the chatbot\n
    """

EVAL_DESC = "#For verbosity, the entire prompt sent to the LLM for answer checking is shown below\n"


ANS_EVAL_PROMPT = PromptTemplate(template=ANS_EVAL_TEMPLATE, input_variables=["query_str","context_str"])