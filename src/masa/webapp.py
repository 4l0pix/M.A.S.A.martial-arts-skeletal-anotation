#local browser api

import argparse
import ipaddress
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import anyio
import cv2
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from masa.pipeline import RunConfig
from masa.video import read_frame
from masa.web_service import ARTIFACT_TYPES, BrowserSession, WebService

POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
    "connect-src 'self'; img-src 'self' blob:; media-src 'self' blob:; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)


def _origin(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path or parsed.query or parsed.fragment):
        raise ValueError("Invalid browser origin.")
    return parsed.scheme, parsed.hostname.lower(), parsed.port or (
        443 if parsed.scheme == "https" else 80
    )


def _trusted_request(request):
    hosts = request.headers.getlist("host")
    if len(hosts) != 1 or any(c in hosts[0] for c in "/\\@?# \t\r\n"):
        return False
    try:
        origin = _origin(f"{request.scope['scheme']}://{hosts[0]}")
        if origin[1] != "localhost" and not ipaddress.ip_address(origin[1]).is_loopback:
            return False
        supplied = request.headers.getlist("origin")
        if len(supplied) > 1 or (supplied and _origin(supplied[0]) != origin):
            return False
    except ValueError:
        return False
    return request.headers.get("sec-fetch-site", "").lower() != "cross-site"


class LocalSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1, max_length=4096)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    model: Literal["yolo11n-pose.pt", "yolo11s-pose.pt"] = "yolo11n-pose.pt"
    tracker: Literal["botsort.yaml", "bytetrack.yaml"] = "botsort.yaml"
    device: Literal["cpu", "mps", "cuda:0"] = "cpu"
    start_s: float = Field(default=0.0, ge=0)
    end_s: float | None = Field(default=None, gt=0)
    stride: int = Field(default=1, ge=1, le=120)
    confidence: float = Field(default=0.25, gt=0, le=1)
    keypoint_confidence: float = Field(default=0.35, gt=0, le=1)
    expected_people: int = Field(default=2, ge=1, le=2)
    rights_note: str = ""
    source_note: str = ""
    selection_mode: Literal["automatic", "manual"] = "automatic"
    fighter_boxes: list[list[float]] | None = Field(default=None, min_length=1, max_length=2)
    fighter_seed_frame: int | None = Field(default=None, ge=0)
    fixed_camera: bool = False


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str
    permission_confirmed: bool
    config: AnalysisConfig = Field(default_factory=AnalysisConfig)


def _png(path, index, filename=None):
    try:
        frame = read_frame(path, index)
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise ValueError("Cannot encode the video frame.")
    except (OSError, ValueError, cv2.error) as error:
        raise HTTPException(404, str(error)) from error
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'} if filename else {}
    return Response(encoded.tobytes(), media_type="image/png", headers=headers)


def create_app(*, cache_dir=None, runs_dir=None, analyzer=None,
               max_upload_bytes=200 * 1024 * 1024):
    service = WebService(cache_dir, runs_dir, analyzer)
    sessions = service.sessions

    @asynccontextmanager
    async def lifespan(app):
        service.enter()
        try:
            yield
        finally:
            await run_in_threadpool(service.leave)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.middleware("http")
    async def browser_guard(request, call_next):
        if not _trusted_request(request):
            response = JSONResponse({"error": "Only same-origin loopback requests are allowed."},
                                    status_code=403)
        elif request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD"):
            session = sessions.get(request.cookies.get("masa_session"))
            token = request.headers.get("x-csrf-token", "")
            if not session or not secrets.compare_digest(
                token.encode(), session.csrf_token.encode()
            ):
                response = JSONResponse(
                    {"error": "Initialize a session and supply its CSRF token."}, status_code=403,
                )
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        if (request.url.path.startswith("/api/") and response.status_code in (400, 416)
                and not response.headers.get("content-type", "").startswith("application/json")):
            #file range failures are emitted directly by starlette, not raised
            iterator = getattr(response, "body_iterator", None)
            if iterator is not None:
                async for _ in iterator:
                    pass
            headers = ({"Content-Range": response.headers["content-range"]}
                       if "content-range" in response.headers else {})
            message = ("Invalid byte range." if response.status_code == 400 else
                       "Requested byte range is outside the file.")
            response = JSONResponse({"error": message}, status_code=response.status_code,
                                    headers=headers)
        response.headers["Content-Security-Policy"] = POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse({"error": str(error.detail)}, status_code=error.status_code,
                            headers=error.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        #do not echo arbitrary inputs or nonfinite floats back into json
        detail = [{"location": list(item["loc"]), "message": item["msg"]}
                  for item in error.errors()]
        return JSONResponse({"error": "Invalid request.", "detail": detail}, status_code=422)

    def current_session(request):
        return service.session(request.cookies.get("masa_session"))

    package = Path(__file__).resolve().parent

    @app.get("/")
    def index():
        return FileResponse(package / "web" / "index.html", media_type="text/html")

    app.mount("/assets", StaticFiles(directory=package / "web"), name="assets")
    app.mount("/fonts", StaticFiles(directory=package / "assets" / "defence-signal"), name="fonts")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/bootstrap")
    def bootstrap(request: Request, response: Response):
        session_id = request.cookies.get("masa_session")
        if session_id not in sessions:
            session_id = secrets.token_urlsafe(32)
            sessions[session_id] = BrowserSession()
            response.set_cookie("masa_session", session_id, httponly=True, samesite="strict")
        session = sessions[session_id]
        defaults = asdict(RunConfig())
        defaults.pop("rights_note")
        defaults.pop("source_note")
        defaults["selection_mode"] = "manual"
        return {
            "csrf_token": session.csrf_token,
            "limits": {"upload_bytes": max_upload_bytes, "interval_s": 120},
            "defaults": defaults,
            "source": session.source.public() if session.source else None,
            "job": session.job.public() if session.job else None,
        }

    @app.post("/api/sources/path")
    def local_source(body: LocalSourceRequest, request: Request):
        return service.load_path(current_session(request), body.path).public()

    @app.post("/api/sources/upload")
    async def upload_source(request: Request, filename: str):
        session = current_session(request)
        service.ensure_source_idle(session)
        if request.headers.get("content-type", "").split(";")[0] != "application/octet-stream":
            raise HTTPException(415, "Send raw video bytes as application/octet-stream.")
        length = request.headers.get("content-length")
        if length is not None:
            if not length.isascii() or not length.isdecimal():
                raise HTTPException(422, "Invalid upload Content-Length.")
            try:
                declared_size = int(length)
            except ValueError as error:
                raise HTTPException(422, "Invalid upload Content-Length.") from error
            if declared_size > max_upload_bytes:
                raise HTTPException(413, "Upload exceeds the byte limit; use a local path instead.")
        name = filename.replace("\\", "/").rsplit("/", 1)[-1]
        path = service.upload_path(name)
        retained = False
        try:
            size = 0
            async with await anyio.open_file(path, "xb") as handle:
                async for block in request.stream():
                    size += len(block)
                    if size > max_upload_bytes:
                        raise HTTPException(
                            413, "Upload exceeds the byte limit; use a local path instead.",
                        )
                    await handle.write(block)
            source = await run_in_threadpool(service.accept_source, session, path, "upload", name)
            retained = source.path == path
            return source.public()
        finally:
            if not retained:
                path.unlink(missing_ok=True)

    @app.get("/api/sources/{source_id}/video")
    def source_video(source_id: str, request: Request):
        source = service.source(current_session(request), source_id)
        if not source.path.is_file():
            raise HTTPException(404, "Source video is unavailable; reload the source.")
        return FileResponse(source.path)

    @app.get("/api/sources/{source_id}/frames/{index}")
    def source_frame(source_id: str, index: int, request: Request, download: bool = False):
        source = service.source(current_session(request), source_id)
        if not source.path.is_file():
            raise HTTPException(404, "Source video is unavailable; reload the source.")
        service.check_source(source)
        filename = f"source-frame-{index:06d}.png" if download else None
        return _png(source.path, index, filename)

    @app.post("/api/jobs", status_code=202)
    def start_job(body: JobRequest, request: Request):
        if body.permission_confirmed is not True:
            raise HTTPException(422, "Explicit footage permission confirmation is required.")
        values = body.config.model_dump()
        if not values["rights_note"].strip():
            values["rights_note"] = (
                "User confirmed permission to process this footage in the local UI; "
                "no additional rights notes supplied."
            )
        return service.start_job(current_session(request), body.source_id, RunConfig(**values))

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        return service.job_public(current_session(request), job_id)

    @app.get("/api/jobs/{job_id}/manifest")
    def get_manifest(job_id: str, request: Request):
        return service.manifest(current_session(request), job_id)

    @app.get("/api/jobs/{job_id}/records")
    def get_records(job_id: str, request: Request):
        return service.records(current_session(request), job_id)

    @app.get("/api/jobs/{job_id}/features")
    def get_features(job_id: str, request: Request):
        return service.features(current_session(request), job_id)

    @app.get("/api/jobs/{job_id}/frames/{ordinal}")
    def review_frame(job_id: str, ordinal: int, request: Request,
                     view: Literal["original", "overlay"] = "overlay"):
        path, index = service.review_frame(current_session(request), job_id, ordinal, view)
        return _png(path, index)

    @app.get("/api/jobs/{job_id}/artifacts/{name}")
    def get_artifact(job_id: str, name: str, request: Request, download: bool = False):
        path = service.artifact(current_session(request), job_id, name)
        return FileResponse(path, media_type=ARTIFACT_TYPES[name], filename=name,
                            content_disposition_type="attachment" if download else "inline")

    return app


def main(argv=None):
    import uvicorn

    parser = argparse.ArgumentParser(description="Local M.A.S.A. browser inspector.")
    parser.add_argument("--port", type=int, default=8501)
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.port <= 65535:
        parser.error("port must be within 1..65535")
    uvicorn.run(create_app(), host="127.0.0.1", port=arguments.port)


if __name__ == "__main__":
    main()
