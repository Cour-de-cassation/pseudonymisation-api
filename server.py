import uvicorn
from app import app # noqa

if __name__ == "__main__":
    from argparse import ArgumentParser

    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "-a", "--address",
        default="0.0.0.0",
        help="Host for the app",
    )

    argument_parser.add_argument(
        "-p", "--port",
        default=8081,
        help="Port for the app",
        type=int,
    )
    argument_parser.add_argument(
        "-d", "--debug",
        default=8081,
        help="Debug mode",
        action="store_true",
    )
    argument_parser.add_argument(
        "-l", "--log-level",
        default="error",
        help="Log level"
    )

    arguments = argument_parser.parse_args()

    uvicorn.run(
        "app:app",
        host=arguments.address,
        port=arguments.port,
        log_level=arguments.log_level,
        reload=arguments.debug,
    )
