

prompt_inyection = """
    Your goal is to answer questions using texts as references. Return the answer to the query and choose only one Id of the text that was more relevant to answer the query.
    Do not answer the query if the information is not found in the text references.
    
    ```TypeScript
    RAG: {
    ANSWER: string // The answer of the query based on the reference texts received. Only used the information of the referenced texts, do not answer if the texts are irrelevant.
    TEXT: number // The id of the most relevant text to answer the query. Return 0 if the query was not answered.
    }
    ```
    Please output the extracted information in Json format. Do not output anything except for the query and the text id. 
    Do not add any fields that are not in the schema, and use the same name for each entity.

    Input: QUESTION: Are dogs better listeners than humans? PASSAGES:\n Text 1: Dogs have a great hearing, they are capable of identifying other dogs based on the barking. \n Text 2: Dogs have different sizes and colors as many other animals. \n Text 3: Humans are unable to hear across long distances, however dogs can listen sounds up to 25 mts. away from their position.
    Output: <json>{"RAG": {"ANSWER": "Yes", "TEXT": 3}}</json>
    
    Input: QUESTION: How many eyes does a spider have? PASSAGES:\n Text 1: There are many spiders in the world \n Text 2: All the spiders have 8 legs and some of them are dangerous \n Text 3: Little kids are oftern afraid of spiders.
    Output: <json>{"RAG": {"ANSWER": "the texts do not contain the answer", "TEXT": -1}}</json>"""

promt_only_query = "Answer the following query: {}, based on the documents you received. Do not answer anything that is not presented in the documents."
