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

ANS_EVAL_PROMPT = PromptTemplate(template=ANS_EVAL_TEMPLATE, input_variables=["query_str","context_str"])