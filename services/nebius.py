import os

from openai import OpenAI


NEBIUS_BASE_URL = (
    "https://api.tokenfactory.nebius.com/v1/"
)

NEBIUS_MODEL = (
    "nvidia/Nemotron-3_5-Lightning"
)


def get_nebius_client():
    """
    Create a Nebius Token Factory client.

    The API key must be stored in the NEBIUS_API_KEY
    environment variable. Never hard-code the key.
    """

    api_key = os.environ.get(
        "NEBIUS_API_KEY"
    )

    if not api_key:
        return None

    return OpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=api_key,
    )


def nebius_available() -> bool:
    """
    Return True when a Nebius API key is available.
    """

    return bool(
        os.environ.get(
            "NEBIUS_API_KEY"
        )
    )


def generate_nebius_response(
    messages: list,
) -> dict:
    """
    Send a chat request to NVIDIA Nemotron 3.5 Lightning
    through Nebius Token Factory.

    If Nebius credentials are not available, return a
    structured response instead of crashing the application.
    """

    client = get_nebius_client()

    if client is None:
        return {
            "status": "nebius_not_configured",
            "model": NEBIUS_MODEL,
            "message": (
                "Nebius Token Factory credentials are not "
                "configured yet. Set NEBIUS_API_KEY when "
                "hackathon access becomes available."
            ),
        }

    response = client.chat.completions.create(
        model=NEBIUS_MODEL,
        messages=messages,
    )

    return {
        "status": "success",
        "model": NEBIUS_MODEL,
        "content": (
            response
            .choices[0]
            .message
            .content
        ),
    }
