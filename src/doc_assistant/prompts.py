"""Prompt templates for the RAG pipeline."""
from langchain_core.prompts import ChatPromptTemplate


MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query expansion assistant. "
        "Generate 3 paraphrases of the given question that preserve its meaning exactly but use different vocabulary or sentence structure."
        
        "Rules; Vary them as follows:\n"
        "1. Do NOT make the question broader or more general.\n"
        "2. Do NOT make the question narrower or more specific.\n"
        "3. Each paraphrase should use distinct vocabulary where possible.\n"
        "4. Same scope, same intent — just different wording.\n\n"
        "Output ONLY a JSON array of strings, nothing else. No explanation, no markdown."
    )),
    ("human", "Original question:\n{question}\n\nJSON array:"),
])


REWRITE_PROMPT = ChatPromptTemplate.from_template("""
Given a chat history and a follow-up question, rewrite the follow-up question 
to be a standalone question that captures the full context.
If the question is already standalone, return it unchanged.
Output ONLY the rewritten question, nothing else.

Chat history:
{history}

Follow-up question: {question}

Standalone question:
""")


ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful research assistant. Answer the question based only on the 
following sources. Cite sources inline using [1], [2], etc. matching their 
numbers below. If the sources don't contain the answer, say so clearly.

Sources:
{context}

Question: {question}

Answer:
""")