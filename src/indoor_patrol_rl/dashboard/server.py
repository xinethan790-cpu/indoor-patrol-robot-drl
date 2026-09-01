"""Loopback-only stdlib HTTP server for the patrol dashboard."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from indoor_patrol_rl.dashboard.projector import DashboardMap, project_dashboard
from indoor_patrol_rl.dashboard.repository import (
    DashboardEvidenceError,
    PatrolDashboardRepository,
)

_LOOPBACK_HOST = "127.0.0.1"
_STATIC_DIRECTORY = Path(__file__).with_name("static")
_STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/static/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
}


class PatrolDashboardServer(ThreadingHTTPServer):
    """HTTP server deliberately restricted to the IPv4 loopback interface."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        repository: PatrolDashboardRepository,
        dashboard_map: DashboardMap,
        *,
        port: int = 8765,
    ) -> None:
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer between 0 and 65535.")
        self.repository = repository
        self.dashboard_map = dashboard_map
        super().__init__((_LOOPBACK_HOST, port), _DashboardRequestHandler)

    def dashboard_payload(self) -> dict[str, object]:
        return project_dashboard(self.repository.read(), self.dashboard_map)


class _DashboardRequestHandler(BaseHTTPRequestHandler):
    server: PatrolDashboardServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self._serve(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(head_only=True)

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_is_allowed():
            self._json_response(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_host"},
                head_only=False,
            )
            return
        self._json_response(
            HTTPStatus.METHOD_NOT_ALLOWED,
            {"error": "method_not_allowed"},
            head_only=False,
            extra_headers={"Allow": "GET, HEAD"},
        )

    do_PUT = do_POST
    do_DELETE = do_POST
    do_PATCH = do_POST

    def _serve(self, *, head_only: bool) -> None:
        if not self._host_is_allowed():
            self._json_response(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_host"},
                head_only=head_only,
            )
            return
        path = unquote(urlsplit(self.path).path)
        if "\\" in path or "\x00" in path or any(part == ".." for part in path.split("/")):
            self._json_response(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found"},
                head_only=head_only,
            )
            return
        if path == "/api/dashboard":
            try:
                payload = self.server.dashboard_payload()
            except DashboardEvidenceError as error:
                self._json_response(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "evidence_unavailable", "message": str(error)},
                    head_only=head_only,
                )
                return
            self._json_response(HTTPStatus.OK, payload, head_only=head_only)
            return
        if path == "/healthz":
            self._json_response(
                HTTPStatus.OK,
                {"status": "ok", "binding": "loopback_only"},
                head_only=head_only,
            )
            return
        static = _STATIC_ROUTES.get(path)
        if static is None:
            self._json_response(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found"},
                head_only=head_only,
            )
            return
        filename, content_type = static
        static_path = _STATIC_DIRECTORY / filename
        try:
            body = static_path.read_bytes()
        except OSError:
            self._json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "static_asset_unavailable"},
                head_only=head_only,
            )
            return
        self._bytes_response(
            HTTPStatus.OK,
            body,
            content_type=content_type,
            cache_control="no-store",
            head_only=head_only,
        )

    def _host_is_allowed(self) -> bool:
        values = self.headers.get_all("Host", failobj=[])
        if len(values) != 1:
            return False
        value = values[0].strip().casefold()
        port = self.server.server_address[1]
        return value in {
            "localhost",
            f"localhost:{port}",
            _LOOPBACK_HOST,
            f"{_LOOPBACK_HOST}:{port}",
        }

    def _json_response(
        self,
        status: HTTPStatus,
        payload: dict[str, object],
        *,
        head_only: bool,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._bytes_response(
            status,
            body,
            content_type="application/json; charset=utf-8",
            cache_control="no-store",
            head_only=head_only,
            extra_headers=extra_headers,
        )

    def _bytes_response(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str,
        cache_control: str,
        head_only: bool,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Keep one concise local access line without exposing evidence contents."""

        print(f"DASHBOARD_HTTP {self.address_string()} {format % args}", flush=True)


def serve_dashboard(
    *,
    run_directory: str | Path,
    map_config: str | Path,
    port: int = 8765,
) -> None:
    """Serve until interrupted, always on ``127.0.0.1``."""

    repository = PatrolDashboardRepository(run_directory)
    dashboard_map = DashboardMap.from_yaml(map_config)
    with PatrolDashboardServer(repository, dashboard_map, port=port) as server:
        actual_port = server.server_address[1]
        print(
            f"巡检看板已启动：http://{_LOOPBACK_HOST}:{actual_port} "
            f"（仅本机可访问，run={repository.run_directory.name}）",
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            print("巡检看板已停止。", flush=True)
