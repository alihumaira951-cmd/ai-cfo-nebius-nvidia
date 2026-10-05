import os

from openai import OpenAI


# ---------------------------------------------------------
# Nebius Token Factory configuration
# ---------------------------------------------------------

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Can be overridden later without changing application code.
NEBIUS_MODEL = os.environ.get(
    "NEBIUS_MODEL",
    "nvidia/Nemotron-3_5-Lightning",
)


# ---------------------------------------------------------
# NVIDIA direct API configuration
# ---------------------------------------------------------

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

NVIDIA_MODEL = os.environ.get(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3-super-120b-a12b",
)


# ---------------------------------------------------------
# Clients
# ---------------------------------------------------------

def get_nebius_client():
    """
    Create a Nebius Token Factory client.

    The API key must be stored in the NEBIUS_API_KEY
    environment variable. Never hard-code the key.
    """

    api_key = os.environ.get("NEBIUS_API_KEY")

    if not api_key:
        return None

    return OpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=api_key,
    )


def get_nvidia_client():
    """
    Create a direct NVIDIA API client.

    The API key must be stored in the NVIDIA_API_KEY
    environment variable. Never hard-code the key.
    """

    api_key = os.environ.get("NVIDIA_API_KEY")

    if not api_key:
        return None

    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
    )


# ---------------------------------------------------------
# Availability checks
# ---------------------------------------------------------

def nebius_available() -> bool:
    """
    Return True when a Nebius API key is available.
    """

    return bool(os.environ.get("NEBIUS_API_KEY"))


def nvidia_available() -> bool:
    """
    Return True when a direct NVIDIA API key is available.
    """

    return bool(os.environ.get("NVIDIA_API_KEY"))


def live_nemotron_available() -> bool:
    """
    Return True when either live Nemotron provider
    is configured.
    """

    return nebius_available() or nvidia_available()


# ---------------------------------------------------------
# Nemotron generation
# ---------------------------------------------------------

def generate_nebius_response(
    messages: list,
) -> dict:
    """
    Generate the AI CFO executive reasoning response.

    Provider priority:

    1. NVIDIA Nemotron through Nebius Token Factory
    2. NVIDIA Nemotron through NVIDIA's direct API
    3. Structured not-configured response

    Nebius remains the preferred hackathon provider.
    NVIDIA direct inference allows the live Nemotron
    reasoning layer to operate while Nebius billing or
    promotional-credit access is pending.
    """

    # -----------------------------------------------------
    # Preferred provider: Nebius Token Factory
    # -----------------------------------------------------

    nebius_client = get_nebius_client()

    if nebius_client is not None:
        try:
            response = nebius_client.chat.completions.create(
                model=NEBIUS_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
            )

            return {
                "status": "success",
                "provider": "Nebius Token Factory",
                "model": NEBIUS_MODEL,
                "content": response.choices[0].message.content,
            }

        except Exception as exc:
            nebius_error = str(exc)

    else:
        nebius_error = None

    # -----------------------------------------------------
    # Secondary provider: NVIDIA direct endpoint
    # -----------------------------------------------------

    nvidia_client = get_nvidia_client()

    if nvidia_client is not None:
        try:
            response = nvidia_client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
            )

            return {
                "status": "success",
                "provider": "NVIDIA API",
                "model": NVIDIA_MODEL,
                "content": response.choices[0].message.content,
            }

        except Exception as exc:
            return {
                "status": "error",
                "provider": "NVIDIA API",
                "model": NVIDIA_MODEL,
                "message": (
                    "Live Nemotron inference could not be completed."
                ),
                "error": str(exc),
                "nebius_error": nebius_error,
            }

    # -----------------------------------------------------
    # No live provider configured
    # -----------------------------------------------------

    return {
        "status": "nemotron_not_configured",
        "provider": "Local fallback",
        "model": None,
        "message": (
            "Live Nemotron credentials are not configured. "
            "Set NEBIUS_API_KEY for Nebius Token Factory or "
            "NVIDIA_API_KEY for the direct NVIDIA endpoint."
        ),
        "nebius_error": nebius_error,
    }
