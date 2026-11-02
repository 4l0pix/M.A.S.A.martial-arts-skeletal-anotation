#session owned sources and analysis state

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import secrets
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from zipfile import ZIP_DEFLATED, ZipFile

from starlette.exceptions import HTTPException

from masa.fighter_selection import validate_selection
from masa.pipeline import FEATURE_COLUMNS, RunConfig, analyze_video
from masa.video import VideoInfo, frame_bounds, inspect_video

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
CORE_ARTIFACTS = (
    "manifest.json", "poses.jsonl", "features.csv", "overlay.mp4", "contact_sheet.jpg",
)
ARTIFACT_TYPES = {
    "manifest.json": "application/json", "poses.jsonl": "application/x-ndjson",
    "features.csv": "text/csv", "overlay.mp4": "video/mp4", "contact_sheet.jpg": "image/jpeg",
    "masa-run.zip": "application/zip",
}


def content_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _feature_value(value, integer=False):
    try:
        number = float(value)
        if math.isfinite(number) and (not integer or number.is_integer()):
            return int(number) if integer else number
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _artifact_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise HTTPException(409, "Run artifact is unreadable; run analysis again.") from error


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite artifact evidence.")
    return number


def _artifact_json(value):
    try:
        return json.loads(value, parse_float=_finite_float, parse_constant=_finite_float)
    except ValueError as error:
        raise HTTPException(409, "Run JSON artifact is invalid; run analysis again.") from error


@dataclass
class Source:
    id: str
    name: str
    kind: str
    path: Path
    sha256: str
    info: VideoInfo

    def public(self):
        info = asdict(self.info)
        info.pop("path")
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "sha256": self.sha256,
            "info": info,
            "video_url": f"/api/sources/{self.id}/video",
        }


@dataclass
class Job:
    id: str
    source: Source
    config: RunConfig
    output_dir: Path
    status: str = "queued"
    progress: float = 0.0
    message: str = "Preparing analysis."
    error: str | None = None
    frame_count: int | None = None

    def public(self):
        return {
            "id": self.id, "source_id": self.source.id, "status": self.status,
            "progress": self.progress, "message": self.message, "error": self.error,
            "config": asdict(self.config), "frame_count": self.frame_count,
        }


@dataclass
class BrowserSession:
    csrf_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    source: Source | None = None
    job: Job | None = None
    jobs: dict[str, Job] = field(default_factory=dict)


class WebService:
    def __init__(self, cache_dir=None, runs_dir=None, analyzer=None):
        self.sessions = {}
        self.lock = RLock()
        cache_root = Path(os.environ.get("MASA_CACHE_DIR") or "~/.cache/masa")
        self.cache_dir = Path(cache_dir if cache_dir is not None else cache_root / "uploads")
        self.cache_dir = self.cache_dir.expanduser().resolve()
        self.upload_dir = None
        self.clients = 0
        self.runs_dir = Path(runs_dir or "runs").expanduser().resolve()
        self.analyzer = analyze_video if analyzer is None else analyzer
        self.executor = None
        self.active_job = None

    def enter(self):
        with self.lock:
            self.clients += 1

    def leave(self):
        with self.lock:
            self.clients -= 1
            if self.clients != 0:
                return
            executor = self.executor
        #finish inference before removing an uploaded source
        if executor is not None:
            executor.shutdown(wait=True)
        with self.lock:
            self.executor = None
            if self.upload_dir is not None:
                self.upload_dir.cleanup()
                self.upload_dir = None

    def upload_path(self, name):
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS or len(name) > 255 or "\0" in name:
            raise HTTPException(422, "Choose an MP4, MOV, AVI, MKV, M4V, or WebM video.")
        with self.lock:
            if self.upload_dir is None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.upload_dir = tempfile.TemporaryDirectory(
                    prefix="masa-web-", dir=self.cache_dir,
                )
            return Path(self.upload_dir.name) / (secrets.token_hex(16) + suffix)

    def session(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise HTTPException(404, "Browser session not found; reload the application.")
            return session

    def source(self, session, source_id):
        with self.lock:
            if session.source is None or session.source.id != source_id:
                raise HTTPException(404, "Source not found in this browser session.")
            return session.source

    def load_path(self, session, value):
        return self.accept_source(session, value, "local")

    def accept_source(self, session, value, kind, name=None):
        self.ensure_source_idle(session)
        try:
            path = Path(value).expanduser().resolve()
            if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file():
                raise ValueError("Choose a regular, supported local video file.")
            info = inspect_video(path)
            digest = content_hash(path)
        except (OSError, ValueError) as error:
            raise HTTPException(422, str(error)) from error
        with self.lock:
            self.ensure_source_idle(session)
            if (session.source is not None and session.source.kind == kind
                    and session.source.sha256 == digest
                    and (kind == "upload" or session.source.path == path)):
                return session.source
            previous = session.source
            session.source = Source(
                secrets.token_hex(16), name or path.name, kind, path, digest, info,
            )
            session.job = None
            if previous is not None and previous.kind == "upload":
                previous.path.unlink(missing_ok=True)
            return session.source

    def ensure_source_idle(self, session):
        with self.lock:
            if session.job is not None and session.job.status in ("queued", "running"):
                raise HTTPException(409, "Analysis is running; wait before switching the source.")

    def job(self, session, job_id):
        with self.lock:
            job = session.jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "Job not found in this browser session.")
            return job

    def job_public(self, session, job_id):
        with self.lock:
            return self.job(session, job_id).public()

    def completed_job(self, session, job_id):
        with self.lock:
            job = self.job(session, job_id)
            if job.status != "completed":
                raise HTTPException(409, "Review is available only after analysis completes.")
            return job

    def manifest(self, session, job_id):
        path = self.artifact(session, job_id, "manifest.json")
        return _artifact_json(_artifact_text(path))

    def artifact(self, session, job_id, name):
        job = self.completed_job(session, job_id)
        if name not in ARTIFACT_TYPES:
            raise HTTPException(404, "Unknown run artifact.")
        path = job.output_dir / name
        try:
            available = path.is_file() and path.resolve().parent == job.output_dir
        except (OSError, RuntimeError):
            available = False
        if not available:
            raise HTTPException(404, "Run artifact is unavailable.")
        return path

    def records(self, session, job_id):
        path = self.artifact(session, job_id, "poses.jsonl")
        return [_artifact_json(line) for line in _artifact_text(path).splitlines() if line.strip()]

    def features(self, session, job_id):
        path = self.artifact(session, job_id, "features.csv")
        rows = [{
            name: _feature_value(raw.get(name), name in ("frame_index", "track_id"))
            for name in FEATURE_COLUMNS
        } for raw in csv.DictReader(io.StringIO(_artifact_text(path)))]
        return {
            "tracks": sorted({row["track_id"] for row in rows if row["track_id"] is not None}),
            "features": [{
                "name": name,
                "unit": "degrees" if name.endswith("_deg") else (
                    "torso lengths / second" if name.endswith("_per_s") else "torso lengths"
                ),
            } for name in FEATURE_COLUMNS[3:]],
            "rows": rows,
        }

    def review_frame(self, session, job_id, ordinal, view):
        job = self.completed_job(session, job_id)
        records = self.records(session, job_id)
        if not 0 <= ordinal < len(records):
            raise HTTPException(404, "Processed frame ordinal is outside the run.")
        if view == "original":
            self.check_source(job.source)
            return job.source.path, records[ordinal]["frame_index"]
        return self.artifact(session, job_id, "overlay.mp4"), ordinal

    def start_job(self, session, source_id, config):
        with self.lock:
            source = self.source(session, source_id)
            if self.active_job is not None:
                raise HTTPException(409, "Another analysis is running; wait for it to finish.")
            try:
                config.validate()
                first_frame, _ = frame_bounds(source.info, config.start_s,
                                             config.end_s, config.stride)
                validate_selection(config.selection_mode, config.expected_people,
                                   config.fighter_boxes, config.fighter_seed_frame,
                                   info=source.info, first_frame=first_frame)
                end_s = source.info.duration_s if config.end_s is None else config.end_s
                if end_s - config.start_s > 120:
                    raise ValueError("Select an interval of at most 120 seconds.")
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            job_id = secrets.token_hex(16)
            job = Job(job_id, source, config, self.runs_dir / job_id)
            previous = session.job
            session.job = job
            session.jobs[job_id] = job
            self.active_job = job
        try:
            self.check_source(source)
            with self.lock:
                if self.executor is None:
                    self.executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="masa-analysis",
                    )
                self.executor.submit(self.run_job, job)
                return job.public()
        except BaseException:
            with self.lock:
                self.active_job = None
                session.job = previous
                session.jobs.pop(job_id, None)
            raise

    def check_source(self, source):
        #local files remain mutable; no source copy or stat-only identity shortcut
        try:
            if content_hash(source.path) != source.sha256:
                raise ValueError(
                    "Source contents changed; reload the source and confirm permission again."
                )
        except (OSError, ValueError) as error:
            raise HTTPException(409, str(error)) from error

    def run_job(self, job):
        def progress(fraction, message):
            with self.lock:
                if math.isfinite(fraction):
                    job.progress = min(1.0, max(job.progress, fraction, 0.0))
                job.message = str(message)

        with self.lock:
            job.status = "running"
            job.message = "Preparing video and loading pose model."
        try:
            self.check_source(job.source)
            result = self.analyzer(job.source.path, job.output_dir, job.config, progress=progress)
            self.check_source(job.source)
            manifest = json.loads((job.output_dir / "manifest.json").read_text())
            if manifest["source_sha256"] != job.source.sha256:
                raise ValueError(
                    "Source changed during analysis; reload it and confirm permission again."
                )
            with ZipFile(job.output_dir / "masa-run.zip", "x", compression=ZIP_DEFLATED) as archive:
                for name in CORE_ARTIFACTS:
                    archive.write(job.output_dir / name, arcname=name)
            with self.lock:
                job.status = "completed"
                job.progress = 1.0
                job.message = "Analysis completed."
                job.frame_count = result.frame_count
        except Exception as error:
            with self.lock:
                job.status = "failed"
                job.error = str(error.detail) if isinstance(error, HTTPException) else str(error)
                job.message = "Analysis failed."
        finally:
            with self.lock:
                self.active_job = None
