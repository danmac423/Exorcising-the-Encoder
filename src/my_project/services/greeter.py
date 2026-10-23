# Import from a sibling module in the same package: relative.
from .formatting import shout


def greet(name: str, loud: bool = False) -> str:
    message = f"Hello, {name}!"
    return shout(message) if loud else message
