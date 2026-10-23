from my_project.config import DEFAULT_NAME
from my_project.services.greeter import greet


def test_greet_default() -> None:
    assert greet(DEFAULT_NAME) == "Hello, world!"


def test_greet_loud() -> None:
    assert greet("Daniel", loud=True) == "HELLO, DANIEL!"
