# Import from elsewhere in the package: absolute (always starts with the package name).
from my_project.config import APP_NAME, DEFAULT_NAME
from my_project.services.greeter import greet


def main() -> None:
    print(f"Starting {APP_NAME}...")
    print(greet(DEFAULT_NAME))
