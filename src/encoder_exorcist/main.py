# Import from elsewhere in the package: absolute (always starts with the package name).
from encoder_exorcist.config import APP_NAME, DEFAULT_NAME
from encoder_exorcist.services.greeter import greet


def main() -> None:
    print(f"Starting {APP_NAME}...")
    print(greet(DEFAULT_NAME))
