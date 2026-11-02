#keep model-library analytics disabled within M.A.S.A. only

import json
import os
import sys
from pathlib import Path


def configure_ultralytics_privacy() -> Path:
    cache = Path(os.environ.get("MASA_CACHE_DIR", str(Path.home() / ".cache" / "masa")))
    config_root = cache.expanduser().resolve() / "ultralytics"
    settings_path = config_root / "Ultralytics" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_root)
    os.environ["YOLO_AUTOINSTALL"] = "false"

    #seed an opt-out before the first vendor import; its manager supplies the remaining schema
    try:
        with settings_path.open("x", encoding="utf-8") as settings_file:
            settings_file.write('{"sync": false}\n')
    except FileExistsError:
        pass

    import ultralytics
    from ultralytics.utils import SettingsManager

    active = ultralytics.settings
    if Path(active.file).resolve() == settings_path:
        active.update({"sync": False})
    else:
        #an already-imported library may belong to another project never save its settings file
        private = SettingsManager(file=settings_path)
        private.update({"sync": False})
        dict.__setitem__(active, "sync", False)

    #existing event collectors cache their enabled flag, so changing settings alone is insufficient
    for name in ("ultralytics.utils.events", "ultralytics.hub.utils"):
        module = sys.modules.get(name)
        collector = getattr(module, "events", None)
        if collector is not None and hasattr(collector, "enabled"):
            collector.enabled = False
            queue = getattr(collector, "events", None)
            if isinstance(queue, list):
                queue.clear()

    persisted_sync = json.loads(settings_path.read_text())["sync"]
    if active.get("sync") is not False or persisted_sync is not False:
        raise RuntimeError("Could not persist the local model-library telemetry opt-out.")
    return settings_path
