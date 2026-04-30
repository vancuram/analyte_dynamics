from __future__ import annotations

import json
from pathlib import Path
import importlib.util
import sys
from plotly.utils import PlotlyJSONEncoder


HERE = Path(__file__).resolve().parent
WEB_DIR = HERE.parent
APP_PATH = WEB_DIR / "app.py"
FIG_DIR = HERE / "figures"
MANIFEST_PATH = HERE / "manifest.json"

if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))


def load_app_module():
    spec = importlib.util.spec_from_file_location("lab_web_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import app module from {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_figure_json(path: Path, fig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = fig.to_plotly_json()
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, cls=PlotlyJSONEncoder)


def main() -> None:
    app = load_app_module()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tests = app.get_available_tests()
    options = app.build_test_options(tests)

    manifest = {
        "tests": [{"id": int(opt["value"]), "label": str(opt["label"])} for opt in options],
        "confounderDisplay": app.CONFOUNDER_DISPLAY,
        "confoundersByTest": {},
        "figureFiles": {},
    }

    for opt in options:
        test_id = int(opt["value"])
        confs = app.get_available_confounders_for_test(test_id)
        manifest["confoundersByTest"][str(test_id)] = confs

        tw = app.create_time_windows_plot(test_id)
        key = f"{test_id}__time_windows"
        rel = f"figures/{key}.json"
        save_figure_json(HERE / rel, tw)
        manifest["figureFiles"][key] = rel

        tw_mobile = app.create_time_windows_plot(test_id, mobile=True)
        key = f"{test_id}__time_windows__mobile"
        rel = f"figures/{key}.json"
        save_figure_json(HERE / rel, tw_mobile)
        manifest["figureFiles"][key] = rel

        ar = app.create_admission_routine_plot(test_id)
        key = f"{test_id}__admission_routine"
        rel = f"figures/{key}.json"
        save_figure_json(HERE / rel, ar)
        manifest["figureFiles"][key] = rel

        qt = app.create_quartiles_plot(test_id)
        key = f"{test_id}__quartiles"
        rel = f"figures/{key}.json"
        save_figure_json(HERE / rel, qt)
        manifest["figureFiles"][key] = rel

        qt_mobile = app.create_quartiles_plot(test_id, mobile=True)
        key = f"{test_id}__quartiles__mobile"
        rel = f"figures/{key}.json"
        save_figure_json(HERE / rel, qt_mobile)
        manifest["figureFiles"][key] = rel

        for conf in confs:
            cf = app.create_confounder_plot(test_id, conf)
            key = f"{test_id}__confounder__{conf}"
            rel = f"figures/{key}.json"
            save_figure_json(HERE / rel, cf)
            manifest["figureFiles"][key] = rel

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Built static site files at: {HERE}")
    print(f"Tests: {len(options)}")
    print(f"Figures: {len(manifest['figureFiles'])}")


if __name__ == "__main__":
    main()
