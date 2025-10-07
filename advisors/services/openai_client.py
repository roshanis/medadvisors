"""Centralised OpenAI client helpers used across the app."""

from functools import lru_cache
from openai import OpenAI


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Return a cached OpenAI client instance.

    The client is lightweight, but creating it repeatedly in hot paths (e.g. for
    every advisor response) adds unnecessary overhead. Caching the client keeps
    session state minimal while ensuring any later environment changes (such as
    setting the API key) occur before the first call into this helper.
    """

    return OpenAI()
