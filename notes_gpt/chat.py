import os
from openai import OpenAI
from retrieve import hybrid_retrieve

MODEL = "openai/gpt-oss-120b"

def get_llm() -> OpenAI:
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )

def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[{c['source']} p.{c['page']}] {c['text']}" for c in chunks)
    return [
        {
            "role": "system",
            "content": (
                "Answer using ONLY the context. Cite sources like [filename p.N]. "
                "If not in context, say 'I don't know'."
            ),
        },
        {"role": "user", "content": f"<context>\n{context}\n</context>\n\nQ: {question}"},
    ]

def answer_streaming(question: str, collection):
    chunks = hybrid_retrieve(question, collection)
    if not chunks:
        yield "I don't know — nothing relevant was found in your uploaded documents."
        return
    messages = build_prompt(question, chunks)
    stream = get_llm().chat.completions.create(model=MODEL, messages=messages, stream=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
