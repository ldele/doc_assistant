"""Prompt templates for the RAG pipeline."""
from langchain_core.prompts import ChatPromptTemplate


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