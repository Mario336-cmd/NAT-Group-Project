# Serve TrueCheck and expose the local claim-verification pipeline.

from pathlib import Path
from threading import Lock, BoundedSemaphore
import asyncio
import os
from contextlib import asynccontextmanager, suppress
import json
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import ValidationError
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .bert_model import ClaimInputError
from .extraction import ExtractionError, TextExtractor
from .settings import Settings
from .speech import SpeechError, SpeechTranscriber
from .verification import PipelineBusyError, PipelineUnavailableError, VerificationService
from .configuration import COOKIE_NAME, PROFILE_COOKIE_NAME, SLOT_PURPOSES, ConfigurationInput, PersistenceError, SessionStore


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "Frontend"
settings = Settings()
service = VerificationService(settings)
extractor = TextExtractor(settings)
transcriber = SpeechTranscriber(settings)
transcription_lock = Lock()
extraction_lock = Lock()
sessions = SessionStore(settings)
report_slots = BoundedSemaphore(2)


async def prepare_models():
    try:
        await run_in_threadpool(service.warmup)
    except Exception:
        # Component states expose a sanitized setup failure through health.
        service._progress(state="error")


@asynccontextmanager
async def lifespan(application):
    async def expire_sessions():
        while True:
            await asyncio.sleep(30)
            sessions.cleanup()

    cleanup = asyncio.create_task(expire_sessions())
    service.queue_warmup()
    application.state.model_task = asyncio.create_task(prepare_models())
    try:
        yield
    finally:
        cleanup.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup
        await application.state.model_task
        sessions.close()

app = FastAPI(
    title="TrueCheck",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts), www_redirect=False)


def same_origin(request):
    try:
        supplied = urlsplit(request.headers.get("origin", ""))
        expected = request.url
        return (supplied.scheme in ("http", "https") and not supplied.username and not supplied.password
                and supplied.path in ("", "/") and not supplied.query and not supplied.fragment
                and (supplied.scheme, supplied.hostname, supplied.port or (443 if supplied.scheme == "https" else 80))
                == (expected.scheme, expected.hostname, expected.port or (443 if expected.scheme == "https" else 80)))
    except ValueError:
        return False


@app.middleware("http")
async def protect_requests(request, call_next):
    if request.url.path.startswith("/api/"):
        if request.url.scheme != "https" and request.url.hostname not in ("localhost", "127.0.0.1", "::1"):
            return JSONResponse({"detail": "HTTPS is required outside localhost."}, status_code=403)
        if "origin" in request.headers and not same_origin(request):
            return JSONResponse({"detail": "Requests must come from this application's origin."}, status_code=403)
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            if not same_origin(request):
                return JSONResponse({"detail": "Requests must come from this application's origin."}, status_code=403)
            session = sessions.authorize(request.cookies.get(COOKIE_NAME), request.headers.get("x-csrf-token"))
            if session is None:
                return JSONResponse({"detail": "Your session expired or could not be verified. Open Settings and try again."}, status_code=403)
            request.state.configuration_session = session
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )
    # Prevent a different app previously served on this port from caching these files.
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(RequestValidationError)
async def sanitized_validation_error(request, error):
    # Pydantic's default response can echo submitted values. Return a generic message.
    return JSONResponse({"detail": "Invalid request fields. Check the form and try again."}, status_code=422)


@app.get("/api/configuration")
def configuration(request: Request, response: Response):
    session = sessions.get(request.cookies.get(COOKIE_NAME))
    if session is None:
        try:
            persistent = (os.name == "nt" and request.url.hostname in ("127.0.0.1", "localhost", "::1")
                          and request.client is not None and request.client.host in ("127.0.0.1", "::1"))
            session = sessions.create(request.cookies.get(PROFILE_COOKIE_NAME), persistent=persistent)
        except RuntimeError:
            raise HTTPException(503, "The session limit has been reached. Please retry later.") from None
        response.set_cookie(COOKIE_NAME, session.token, httponly=True, samesite="strict",
                            secure=request.url.scheme == "https", path="/")
    if session.profile_id is not None:
        # An opaque browser-profile token restores only this browser's credentials.
        response.set_cookie(PROFILE_COOKIE_NAME, session.profile_id, max_age=365 * 24 * 60 * 60,
                            httponly=True, samesite="strict", secure=request.url.scheme == "https", path="/")
    return sessions.status(session)


@app.put("/api/configuration/{slot}")
async def save_configuration(slot: int, request: Request):
    if slot not in SLOT_PURPOSES:
        raise HTTPException(404, "Choose configuration 1, 2 or 3.")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "Settings must be sent as JSON.")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 8192:
            raise HTTPException(413, "The settings request is too large.")
        body.extend(chunk)
    try:
        supplied = ConfigurationInput.model_validate(json.loads(body))
    except (ValidationError, ValueError, UnicodeError):
        raise HTTPException(422, "Check the provider, purpose, model name and API key format.") from None
    session = request.state.configuration_session
    try:
        sessions.save(session, slot, supplied)
        return sessions.status(session)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except PersistenceError as exc:
        raise HTTPException(503, str(exc)) from None


@app.delete("/api/configuration/{slot}")
def clear_configuration(slot: int, request: Request):
    if slot not in SLOT_PURPOSES:
        raise HTTPException(404, "Choose configuration 1, 2 or 3.")
    session = request.state.configuration_session
    try:
        sessions.clear(session, slot)
    except PersistenceError as exc:
        raise HTTPException(503, str(exc)) from None
    return sessions.status(session)


@app.delete("/api/configuration")
def clear_all_configuration(request: Request, response: Response):
    try:
        sessions.clear(request.state.configuration_session)
    except PersistenceError as exc:
        raise HTTPException(503, str(exc)) from None
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="strict",
                           secure=request.url.scheme == "https")
    response.delete_cookie(PROFILE_COOKIE_NAME, path="/", httponly=True, samesite="strict",
                           secure=request.url.scheme == "https")
    return {"cleared": True}


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1, max_length=settings.max_claim_characters, strict=True)
    top_k: int = Field(default=settings.top_k, ge=1, le=10, strict=True)

    @field_validator("claim")
    @classmethod
    def nonempty_claim(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter a claim first.")
        return value.strip()


@app.get("/api/health")
def health(request: Request, response: Response) -> dict:
    # Report local asset readiness without loading the models or calling an API.
    response.headers["Cache-Control"] = "no-store"
    missing = settings.asset_issues()
    available = not missing
    session = sessions.get(request.cookies.get(COOKIE_NAME))
    explanation = sessions.for_purpose(session, "explanation")
    extraction = sessions.for_purpose(session, "extraction")
    transcription = sessions.for_purpose(session, "transcription")
    startup = service.initialization_status()
    return {
        "status": "ok", "verification_available": available,
        "pipeline_status": startup["status"], "components": startup["components"],
        "progress": startup["progress"],
        "explanation_available": explanation is not None,
        "explanation_provider": explanation.provider if explanation else None,
        "explanation_model": explanation.model if explanation else None,
        "extraction_available": extraction is not None,
        "extraction_provider": extraction.provider if extraction else None,
        "extraction_model": extraction.model if extraction else None,
        "transcription_available": transcription is not None,
        "transcription_model": transcription.model if transcription else None,
        "message": (
            "Local model or index files are missing. Restore the saved application assets."
            if missing else (
                "Local setup could not finish. Choose Retry to try again."
                if startup["status"] == "error" else
                "Preparing the retriever and classifier."
                if startup["status"] in ("not_loaded", "loading") else "Local models are ready."
            )
        ),
    }


@app.post("/api/initialize", status_code=202)
async def initialize_models():
    if settings.asset_issues():
        raise HTTPException(503, "Local model or index files are missing. Restore the saved application assets, then retry.")
    task = getattr(app.state, "model_task", None)
    if service.state != "ready" and (task is None or task.done()):
        service.queue_warmup()
        app.state.model_task = asyncio.create_task(prepare_models())
    return service.initialization_status()


@app.post("/api/export/{report_format}")
async def export_report(report_format: str, request: Request):
    if report_format not in ("pdf", "docx"):
        raise HTTPException(404, "Choose PDF or Word as the report format.")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "Report data must be sent as JSON.")
    if not report_slots.acquire(blocking=False):
        raise HTTPException(409, "Other reports are being prepared. Please retry shortly.")
    try:
        from .reports import ReportRecord, pdf_report, word_report, filename
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > 4 * 1024 * 1024:
                raise HTTPException(413, "This saved result is too large to export.")
            body.extend(chunk)
        try:
            record = ReportRecord.model_validate_json(body)
        except (ValidationError, ValueError):
            raise HTTPException(422, "The saved verification data is incomplete or invalid. Reopen the result and try again.") from None
        builder = pdf_report if report_format == "pdf" else word_report
        data = await run_in_threadpool(builder, record)
        media_type = "application/pdf" if report_format == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        return Response(data, media_type=media_type, headers={"Content-Disposition": 'attachment; filename="' + filename(record, report_format) + '"'})
    except ImportError:
        raise HTTPException(503, "Install the updated backend requirements to enable PDF and Word downloads.") from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, "The report could not be generated. Please retry.") from None
    finally:
        report_slots.release()


@app.post("/api/verify")
def verify(payload: ClaimRequest, request: Request, response: Response) -> dict:
    response.headers["Cache-Control"] = "no-store"
    if settings.asset_issues():
        raise HTTPException(503, "Local model or index files are missing. Check the server setup.")
    try:
        config = sessions.for_purpose(request.state.configuration_session, "explanation")
        return service.verify(payload.claim, top_k=payload.top_k, configuration=config)
    except PipelineBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except PipelineUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ClaimInputError:
        raise HTTPException(422, "The claim could not be encoded. Use a shorter, nonempty text claim.") from None
    except Exception:
        raise HTTPException(503, "Local verification failed. Check the server setup and retry.") from None


@app.post("/api/extract")
async def extract(request: Request, response: Response) -> dict:
    # Bound an upload before parsing it, then extract text without retaining it.
    response.headers["Cache-Control"] = "no-store"
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise HTTPException(415, "Upload one PNG, JPEG, WebP image or PDF using the file field.")
    if not extraction_lock.acquire(blocking=False):
        raise HTTPException(409, "Another file is being read. Please wait and retry.")
    try:
        # Limit the complete request, including multipart headers. Do not trust
        # Content-Length: chunked uploads must obey the same bound.
        request_limit = settings.max_upload_bytes + 64 * 1024
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > request_limit:
                raise HTTPException(413, "Choose a file no larger than 10 MiB.")
            body.extend(chunk)

        async def receive():
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        bounded = Request(request.scope, receive=receive)
        try:
            async with bounded.form(max_files=1, max_fields=0, max_part_size=request_limit) as form:
                items = form.multi_items()
                if len(items) != 1 or items[0][0] != "file" or not isinstance(items[0][1], UploadFile):
                    raise HTTPException(422, "Choose one image or PDF in the file field.")
                upload = items[0][1]
                data = await upload.read(settings.max_upload_bytes + 1)
                if len(data) > settings.max_upload_bytes:
                    raise HTTPException(413, "Choose a file no larger than 10 MiB.")
                filename = upload.filename or ""
                media_type = upload.content_type or ""
            # The multipart context closes any temporary upload spool before
            # sending the validated in-memory file to the selected provider.
            config = sessions.for_purpose(request.state.configuration_session, "extraction")
            return await run_in_threadpool(extractor.extract, data, filename, media_type, configuration=config)
        except StarletteHTTPException as exc:
            if exc.status_code == 400:
                raise HTTPException(422, "The upload could not be read. Choose one image or PDF and retry.") from None
            raise
        except (ValueError, KeyError):
            raise HTTPException(422, "The upload could not be read. Choose one image or PDF and retry.") from None
        except ExtractionError as exc:
            raise HTTPException(exc.status_code, exc.detail) from None
        except Exception:
            raise HTTPException(503, "Text extraction failed. Retry the upload or type your claim instead.") from None
    finally:
        extraction_lock.release()


@app.post("/api/transcriptions")
async def transcribe_audio(request: Request):
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise HTTPException(415, "Send one browser recording in the audio field.")
    if not transcription_lock.acquire(blocking=False):
        raise HTTPException(409, "Another recording is being transcribed. Please wait and retry.")
    try:
        limit = settings.max_audio_bytes + 64 * 1024
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > limit:
                raise HTTPException(413, "Keep recordings to 30 seconds and 5 MiB or fewer.")
            body.extend(chunk)

        async def receive():
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        bounded = Request(request.scope, receive=receive)
        try:
            async with bounded.form(max_files=1, max_fields=0, max_part_size=limit) as form:
                items = form.multi_items()
                if len(items) != 1 or items[0][0] != "audio" or not isinstance(items[0][1], UploadFile):
                    raise HTTPException(422, "Send exactly one recording in the audio field.")
                upload = items[0][1]
                data = await upload.read(settings.max_audio_bytes + 1)
                media_type = upload.content_type or ""
            config = sessions.for_purpose(request.state.configuration_session, "transcription")
            return await run_in_threadpool(transcriber.transcribe, data, media_type, configuration=config)
        except SpeechError as exc:
            raise HTTPException(exc.status_code, exc.detail) from None
        except StarletteHTTPException as exc:
            if exc.status_code == 400:
                raise HTTPException(422, "The recording could not be read. Record it again.") from None
            raise
        except (ValueError, KeyError):
            raise HTTPException(422, "The recording could not be read. Record it again.") from None
        except Exception:
            raise HTTPException(503, "Transcription failed. Your draft was kept. Try again or type your claim.") from None
    finally:
        transcription_lock.release()


# Only frontend files are public; datasets, models and backend files stay outside
# the served directory. Register API routes before this root mount.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
