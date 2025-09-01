import streamlit as st
from groq import Groq
from typing import List, Tuple, Optional

SYSTEM_PROMPT = """You are an AI Career Mentor.
You DO NOT chit-chat. You ONLY produce structured, actionable career guidance.

Always output in exactly this format:

# Career Roadmap
## Overview
- One or two lines describing the target role and context.

## Required Skills
- Core skills
- Tools/technologies
- Optional nice-to-haves

## Learning Path (Step-by-step)
1. ...
2. ...
3. ...
4. ...

## Projects to Build
- Project 1: ...
- Project 2: ...
- Project 3: ...

## Resources
- Course/Doc/Playlist: Name — short why it helps
- ...

## Next Steps (30/60/90 days)
- 30 days: ...
- 60 days: ...
- 90 days: ...
"""

def generate_roadmap(query: str, retrieved: Optional[List[Tuple[str, float]]] = None) -> str:
    """
    Generate a career roadmap using Groq LLM.
    If retrieved knowledge is provided, use it (RAG mode).
    Otherwise, generate purely from the model.
    """
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    if retrieved:
        context_block = "\n\n".join([f"- {t}" for t, _ in retrieved])
        user_prompt = f"""User goal: {query}

Use ONLY the following retrieved knowledge to craft the roadmap (do not invent facts outside this):
{context_block}
"""
    else:
        user_prompt = f"""User goal: {query}

Craft a complete roadmap based on your own knowledge.
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # ✅ updated model
        temperature=0.4,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content
