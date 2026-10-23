from encoder_exorcist.config import DEFAULT_NAME
from encoder_exorcist.services.greeter import greet


def test_greet_default() -> None:
    assert greet(DEFAULT_NAME) == "Hello, world!"


def test_greet_loud() -> None:
    assert greet("Daniel", loud=True) == "HELLO, DANIEL!"
