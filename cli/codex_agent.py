from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


class CodexAppServerError(RuntimeError):
    pass


@dataclass
class TurnTracker:
    chunks: list[str] = field(default_factory=list)
    final_text: str = ""
    error_text: str | None = None
    completed: threading.Event = field(default_factory=threading.Event)


class CodexAppServerSession:
    def __init__(
        self,
        *,
        server_command: str,
        connect_timeout_sec: int = 15,
        request_timeout_sec: int = 60,
        turn_timeout_sec: int = 180,
        extra_env: dict[str, str] | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.server_command = server_command
        self.connect_timeout_sec = connect_timeout_sec
        self.request_timeout_sec = request_timeout_sec
        self.turn_timeout_sec = turn_timeout_sec
        self.extra_env = extra_env or {}
        self._event_callback = event_callback

        self._proc: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_request_id = 1
        self._pending_responses: dict[int, queue.Queue[dict[str, Any]]] = {}

        self._turn_lock = threading.Lock()
        self._turn_trackers: dict[str, TurnTracker] = {}

        self._stop_event = threading.Event()
        self.thread_id: str | None = None
        self.initialized = False

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        try:
            self._event_callback(event_type, payload)
        except Exception:
            # Keep transport alive even if UI callback fails.
            return

    def start(self) -> None:
        if self.running:
            return

        cmd = shlex.split(self.server_command)
        if not cmd:
            raise CodexAppServerError("Empty server command.")

        try:
            env = os.environ.copy()
            env.update(self.extra_env)
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except FileNotFoundError as exc:
            raise CodexAppServerError(f"Failed to start app-server: {exc}") from exc

        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            self.stop()
            raise CodexAppServerError("Failed to open app-server stdio pipes.")

        self._stop_event.clear()
        self._stdout_thread = threading.Thread(target=self._stdout_loop, name="codex-app-server-stdout", daemon=True)
        self._stderr_thread = threading.Thread(target=self._stderr_loop, name="codex-app-server-stderr", daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

        try:
            self.request(
                "initialize",
                {"clientInfo": {"name": "benchmark-cli", "version": "0.1.0"}},
                timeout=self.connect_timeout_sec,
            )
            self.initialized = True
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self._stop_event.set()
        proc = self._proc
        self._proc = None

        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)

        self.initialized = False
        self.thread_id = None
        with self._pending_lock:
            self._pending_responses.clear()
        with self._turn_lock:
            self._turn_trackers.clear()

    def _stdout_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for raw_line in self._proc.stdout:
            if self._stop_event.is_set():
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self._emit("protocol_error", {"line": line})
                continue

            if isinstance(payload, dict) and "id" in payload:
                req_id = payload.get("id")
                if isinstance(req_id, int):
                    with self._pending_lock:
                        response_q = self._pending_responses.pop(req_id, None)
                    if response_q is not None:
                        response_q.put(payload)
                        continue
                self._emit("orphan_response", {"payload": payload})
                continue

            if isinstance(payload, dict) and "method" in payload:
                method = str(payload.get("method", ""))
                params = payload.get("params", {})
                self._handle_notification(method, params if isinstance(params, dict) else {})
                continue

            self._emit("protocol_error", {"line": line})

    def _stderr_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stderr is not None
        for raw_line in self._proc.stderr:
            if self._stop_event.is_set():
                break
            line = raw_line.rstrip("\n")
            if not line:
                continue
            self._emit("stderr", {"line": line})

    def _get_turn_tracker(self, turn_id: str) -> TurnTracker:
        with self._turn_lock:
            tracker = self._turn_trackers.get(turn_id)
            if tracker is None:
                tracker = TurnTracker()
                self._turn_trackers[turn_id] = tracker
            return tracker

    def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        if method == "thread/started":
            thread = params.get("thread", {})
            if isinstance(thread, dict) and isinstance(thread.get("id"), str):
                self.thread_id = thread["id"]

        turn_id: str | None = None
        if isinstance(params.get("turnId"), str):
            turn_id = params["turnId"]
        elif method == "error":
            turn = params.get("turn")
            if isinstance(turn, dict) and isinstance(turn.get("id"), str):
                turn_id = turn["id"]

        if method == "item/agentMessage/delta" and turn_id:
            delta = params.get("delta")
            if isinstance(delta, str):
                self._get_turn_tracker(turn_id).chunks.append(delta)
        elif method == "item/completed" and turn_id:
            item = params.get("item", {})
            if isinstance(item, dict) and item.get("type") == "agentMessage":
                text = item.get("text")
                if isinstance(text, str):
                    self._get_turn_tracker(turn_id).final_text = text
        elif method == "error" and turn_id:
            err_obj = params.get("error")
            if isinstance(err_obj, dict):
                msg = err_obj.get("message")
                if isinstance(msg, str) and msg.strip():
                    self._get_turn_tracker(turn_id).error_text = msg.strip()
                else:
                    self._get_turn_tracker(turn_id).error_text = json.dumps(err_obj, ensure_ascii=True)
            elif err_obj is not None:
                self._get_turn_tracker(turn_id).error_text = str(err_obj)
        elif method == "turn/completed":
            turn = params.get("turn", {})
            if isinstance(turn, dict) and isinstance(turn.get("id"), str):
                turn_id = turn["id"]
            if turn_id:
                self._get_turn_tracker(turn_id).completed.set()

        self._emit("notification", {"method": method, "params": params})

    def request(self, method: str, params: dict[str, Any] | None = None, timeout: int | None = None) -> Any:
        if not self.running:
            raise CodexAppServerError("App-server is not running.")
        if self._proc is None or self._proc.stdin is None:
            raise CodexAppServerError("App-server stdin is unavailable.")

        with self._pending_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            response_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._pending_responses[request_id] = response_q

        request_obj: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            request_obj["params"] = params

        with self._write_lock:
            self._proc.stdin.write(json.dumps(request_obj) + "\n")
            self._proc.stdin.flush()

        wait_timeout = timeout if timeout is not None else self.request_timeout_sec
        try:
            response = response_q.get(timeout=wait_timeout)
        except queue.Empty as exc:
            with self._pending_lock:
                self._pending_responses.pop(request_id, None)
            raise CodexAppServerError(f"Timed out waiting for '{method}' response.") from exc

        if isinstance(response.get("error"), dict):
            error = response["error"]
            raise CodexAppServerError(f"{method} failed: {error}")
        return response.get("result")

    def read_account(self, refresh_token: bool = False) -> dict[str, Any]:
        result = self.request("account/read", {"refreshToken": refresh_token})
        return result if isinstance(result, dict) else {}

    def start_login(self, login_type: str = "chatgpt") -> dict[str, Any]:
        result = self.request("account/login/start", {"type": login_type})
        return result if isinstance(result, dict) else {}

    def cancel_login(self, login_id: str) -> dict[str, Any]:
        result = self.request("account/login/cancel", {"loginId": login_id})
        return result if isinstance(result, dict) else {}

    def logout(self) -> dict[str, Any]:
        result = self.request("account/logout", None)
        return result if isinstance(result, dict) else {}

    def read_rate_limits(self) -> dict[str, Any]:
        result = self.request("account/rateLimits/read", None)
        return result if isinstance(result, dict) else {}

    def ensure_thread(self) -> str:
        if self.thread_id:
            return self.thread_id
        result = self.request("thread/start", {})
        if not isinstance(result, dict):
            raise CodexAppServerError("thread/start returned an unexpected payload.")
        thread = result.get("thread", {})
        if not isinstance(thread, dict) or not isinstance(thread.get("id"), str):
            raise CodexAppServerError("thread/start did not return a thread id.")
        self.thread_id = thread["id"]
        return self.thread_id

    def run_turn(
        self,
        *,
        text: str,
        output_schema: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        thread_id = self.ensure_thread()
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": text}],
        }
        if output_schema is not None:
            params["outputSchema"] = output_schema

        result = self.request("turn/start", params)
        if not isinstance(result, dict):
            raise CodexAppServerError("turn/start returned an unexpected payload.")
        turn = result.get("turn", {})
        if not isinstance(turn, dict) or not isinstance(turn.get("id"), str):
            raise CodexAppServerError("turn/start did not return a turn id.")

        turn_id = turn["id"]
        tracker = self._get_turn_tracker(turn_id)

        wait_timeout = timeout if timeout is not None else self.turn_timeout_sec
        done = tracker.completed.wait(timeout=wait_timeout)
        if not done:
            text_out = tracker.final_text or "".join(tracker.chunks)
            error_text = tracker.error_text or ""
            with self._turn_lock:
                self._turn_trackers.pop(turn_id, None)
            return {
                "threadId": thread_id,
                "turnId": turn_id,
                "text": text_out.strip(),
                "timedOut": True,
                "error": error_text,
            }

        text_out = tracker.final_text or "".join(tracker.chunks)
        error_text = tracker.error_text or ""
        with self._turn_lock:
            self._turn_trackers.pop(turn_id, None)
        return {
            "threadId": thread_id,
            "turnId": turn_id,
            "text": text_out.strip(),
            "timedOut": False,
            "error": error_text,
        }
