"""Prompt templates for the RAG pipeline."""

from langchain_core.prompts import ChatPromptTemplate

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a query expansion assistant. "
                "Generate 3 paraphrases of the given question that preserve "
                "its meaning exactly but use different vocabulary or "
                "sentence structure."
                "Rules; Vary them as follows:\n"
                "1. Do NOT make the question broader or more general.\n"
                "2. Do NOT make the question narrower or more specific.\n"
                "3. Each paraphrase should use distinct vocabulary where possible.\n"
                "4. Same scope, same intent — just different wording.\n\n"
                "Output ONLY a JSON array of strings, nothing else. No explanation, no markdown."
            ),
        ),
        ("human", "Original question:\n{question}\n\nJSON array:"),
    ]
)


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
You are a helpful research assistant. Answer the question using ONLY the sources
below — do not add facts, names, or numbers from outside them.

Citing — both rules matter equally:
- Cite EVERY substantive claim with its source number in square brackets — [1], [3],
  etc. (the source headed "[Source 3: ...]" is cited as [3]). A claim with no [n]
  citation is treated as unsupported, so cite as you write.
- When you cite, use ONLY those bracketed numbers — never an author name, year, key
  (e.g. [Smith2020]), or file name (e.g. (paper.pdf)).
- Do not state any figure, percentage, count, date, or complexity/scaling claim
  (e.g. "reduces it from O(n^2) to O(n)") unless it appears in the sources.
- If the sources don't contain the answer, say so clearly rather than guessing.

Sources:
{context}

Question: {question}

Answer:
""")
