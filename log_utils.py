from fastapi import FastAPI, Response, Request
from starlette.background import BackgroundTask
from starlette.types import Message
import logging
import json
import sys
from datetime import datetime, timezone
from utils import get_juritools_info


async def set_body(request: Request, body: bytes):
    """Utility function to recreate the body of a request"""

    async def receive() -> Message:
        return {"type": "http.request", "body": body}

    request._receive = receive


def disable_loggers():
    """Disable UVICORN and FASTAPI loggers by setting them to CRITICAL levels"""
    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_error.setLevel(level=logging.CRITICAL + 1)
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.setLevel(level=logging.CRITICAL + 1)
    uvicorn_access = logging.getLogger("uvicorn")
    uvicorn_access.setLevel(level=logging.CRITICAL + 1)
    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.setLevel(level=logging.CRITICAL + 1)


def add_custom_logger(
    app: FastAPI,
    custom_logger,
    custom_error_logger,
    disable_uvicorn_logging: bool = True,
) -> FastAPI:
    """Function to add custom loggers to a FastAPI application

    Args:
        app (FastAPI): a FastAPI application
        custom_logger (callable, optional): function used to print logs when working normally.
            Defaults to `default_logger`.
        custom_error_logger (callable, optional): funtion used to print logs when an error occurs.
            Defaults to `default_error_logger`.
        disable_uvicorn_logging (bool, optional): if True, usual uvicorn and FastAPI logs are inhibited.
            Defaults to True.


    Returns:
        FastAPI: FastAPI app with the custom loggers
    """
    if disable_uvicorn_logging:
        disable_loggers()

    @app.middleware("http")
    async def middleware_logger(request: Request, call_next):
        request_body = await request.body()
        await set_body(request, request_body)
        try:
            response = await call_next(request)
        except Exception as exc:
            custom_error_logger(
                **{
                    "request_body": request_body.decode("utf-8"),
                    "request_headers": dict(request.headers),
                    "request_query_params": dict(request.query_params),
                    "request_method": request.method,
                    "request_url": str(request.url),
                    "error_message": str(exc),
                },
            )
            raise exc

        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        task = BackgroundTask(
            custom_logger,
            **{
                "request_body": request_body.decode("utf-8"),
                "request_headers": dict(request.headers),
                "request_query_params": dict(request.query_params),
                "request_method": request.method,
                "request_url": str(request.url),
                "response_body": response_body.decode("utf-8"),
                "response_headers": dict(response.headers),
                "response_media_type": response.media_type,
                "response_status_code": response.status_code,
            },
        )
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
            background=task,
        )

    return app


formatter = logging.Formatter("%(message)s")

logger = logging.getLogger("nlp-api")
standard_output_handler = logging.StreamHandler(stream=sys.stdout)
standard_output_handler.setFormatter(formatter)
logger.addHandler(standard_output_handler)
logger.propagate = False
logger.setLevel(logging.INFO)


def log_error(
    error_message: str,
    request_url: str,
    request_query_params: dict,
    request_method: str,
    request_body: str,
    **kwargs,
):
    """Logs error in JSON format"""
    try:
        data = json.loads(request_body)
    except json.JSONDecodeError:
        data = {}

    date, version = get_juritools_info()

    request_handler = "/" + request_url.split("/", 3)[-1]
    message = {
        "operationName": "NLP-API",
        "msg": "error",
        "data": {
            "error_message": error_message,
            "datetime": str(datetime.now(timezone.utc)),
            "juritools-info": {"date": date, "version": version},
            "request_method": request_method,
            "request_handler": request_handler,
            "request_query_params": request_query_params,
            "idLabel": data.get("idLabel"),
            "idDecision": data.get("idDecision"),
            "sourceId": data.get("sourceId"),
            "sourceName": data.get("sourceName"),
        },
    }

    logger.error(json.dumps(message))


def log_trace(
    request_url: str,
    request_query_params: dict,
    request_method: str,
    request_body: str,
    response_status_code: str,
    **kwargs,
):
    try:
        data = json.loads(request_body)
    except json.JSONDecodeError:
        data = {}

    date, version = get_juritools_info()

    request_handler = "/" + request_url.split("/", 3)[-1]
    message = {
        "operationName": "NLP-API",
        "msg": "trace",
        "data": {
            "response_status_code": response_status_code,
            "datetime": str(datetime.now(timezone.utc)),
            "juritools-info": {"date": date, "version": version},
            "request_method": request_method,
            "request_handler": request_handler,
            "request_query_params": request_query_params,
            "idLabel": data.get("idLabel"),
            "idDecision": data.get("idDecision"),
            "sourceId": data.get("sourceId"),
            "sourceName": data.get("sourceName"),
        },
    }

    logger.info(json.dumps(message))


def log_on_startup():
    date, version = get_juritools_info()

    logger.info(
        json.dumps(
            {
                "operationName": "NLP-API",
                "msg": "Starting server",
                "data": {
                    "datetime": str(datetime.now(timezone.utc)),
                    "juritools-info": {"date": date, "version": version},
                },
            }
        )
    )


def log_on_shutdown():
    date, version = get_juritools_info()

    logger.info(
        json.dumps(
            {
                "operationName": "NLP-API",
                "msg": "Shutting down server",
                "data": {
                    "datetime": str(datetime.now(timezone.utc)),
                    "juritools-info": {"date": date, "version": version},
                },
            }
        )
    )
