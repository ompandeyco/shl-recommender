"""
schemas.py
----------
Pydantic v2 models that define the public API contract for the SHL
Conversational Assessment Recommender.

All request/response shapes used by the FastAPI routes live here so
that validation, serialisation, and OpenAPI documentation are generated
automatically.

Models
------
Message
    A single turn in a conversation: either a "user" or "assistant" message.

ChatRequest
    The payload sent by the client to POST /chat.  Contains the full
    conversation history so the service remains stateless.

Recommendation
    A single SHL assessment result returned to the client.

ChatResponse
    The payload returned by POST /chat.  Carries the assistant's reply,
    zero-or-more assessment recommendations, and a flag signalling whether
    the conversation is over.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    """One turn in the conversation."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """
    Request body for POST /chat.

    Attributes
    ----------
    messages:
        Full conversation history, oldest first.  The last message is
        assumed to be the most recent user turn.
    """

    messages: list[Message]


class Recommendation(BaseModel):
    """
    A single SHL assessment recommended to the user.

    Attributes
    ----------
    name:
        Human-readable assessment name (e.g. "Verify Numerical Reasoning").
    url:
        Canonical product page URL on shl.com.
    test_type:
        Category label (e.g. "Ability & Aptitude", "Personality & Behaviour").
    """

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """
    Response body for POST /chat.

    Attributes
    ----------
    reply:
        The assistant's natural-language message to show the user.
    recommendations:
        List of matched SHL assessments (may be empty while clarifying).
    end_of_conversation:
        True when the agent has finished and no further turns are expected.
    """

    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool
