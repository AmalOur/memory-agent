"""Utility functions used in our graph."""

def split_model_and_provider(fully_specified_name: str) -> dict:
    """Split provider/model string into parts."""
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return {"model": model, "provider": provider}
