import threading
import yaml
import logging
import os
from typing import Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class EdgeGraphConfigLoader:
    _observers = {}  # class-level: path -> observer

    def __init__(self, config_path: str = "config/edge_graph.yaml"):
        self._config_path = config_path
        self._lock = threading.RLock()
        self._config: Dict[str, Any] = {}
        self._load_config()
        # Only start watcher in main process (not in reloader subprocess)
        run_main = os.environ.get("RUN_MAIN")
        if run_main is None or run_main == "true":
            self._start_watcher()

    def _validate_and_normalize(self, config):
        def parse_weights(d):
            if not isinstance(d, dict):
                raise ValueError("Edge types must be a dict of {edge_type: weight}")
            out = {}
            for k, v in d.items():
                try:
                    w = float(v)
                    if w < 0:
                        raise ValueError
                    out[k] = w
                except Exception:
                    raise ValueError(f"Invalid weight for edge type '{k}': {v}")
            return out

        out = {}
        out["default_edge_types"] = parse_weights(config.get("default_edge_types", {}))
        out["app_overrides"] = {}
        for app, edges in (config.get("app_overrides") or {}).items():
            out["app_overrides"][app] = parse_weights(edges)
        return out

    def _load_config(self):
        try:
            with open(self._config_path, "r") as f:
                config = yaml.safe_load(f)
            config = config or {}
            config = self._validate_and_normalize(config)
            with self._lock:
                self._config = config
            logger.info(f"Edge-graph config loaded from {self._config_path}")
        except Exception as e:
            logger.error(f"Failed to load edge-graph config: {e}")

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._config)

    def get_app_edge_weights(self, app_name: str) -> Dict[str, float]:
        with self._lock:
            app_overrides = self._config.get("app_overrides", {})
            if app_name and app_name in app_overrides:
                # Merge: app overrides take precedence, fallback to default
                merged = dict(self._config.get("default_edge_types", {}))
                merged.update(app_overrides[app_name])
                return merged
            return dict(self._config.get("default_edge_types", {}))

    def _start_watcher(self):
        if self._config_path in EdgeGraphConfigLoader._observers:
            # Already watching this path
            return

        class Handler(FileSystemEventHandler):
            def __init__(self, loader):
                self.loader = loader

            def on_modified(self, event):
                if os.path.abspath(event.src_path) == os.path.abspath(self.loader._config_path):
                    logger.info("Edge-graph config file changed, reloading...")
                    self.loader._load_config()

        observer = Observer()
        dirname = os.path.dirname(os.path.abspath(self._config_path)) or "."
        handler = Handler(self)
        observer.schedule(handler, dirname, recursive=False)
        observer.daemon = True
        observer.start()
        EdgeGraphConfigLoader._observers[self._config_path] = observer
