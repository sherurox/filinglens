"""Label Studio REST API: create the risk-paragraph project, import tasks, export (FL-M2 T4).

Label Studio runs locally in Docker Compose (service `labelstudio`, http://127.0.0.1:58080). Its
admin user and API token come from .env, so setup needs no clicking in the UI and no token is
copied by hand. Authentication uses the token header "Authorization: Token <token>".

Commands (from the repo root, after `set -a; source .env; set +a`):
    uv run python -m filinglens.extract.label_studio setup-risk   # create the project, import tasks
    uv run python -m filinglens.extract.label_studio export       # all tasks + labels, as a backup

setup-risk is idempotent: it finds the project by title and imports only into an empty project,
so running it twice never duplicates tasks.
"""

import argparse
import html
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

DEFAULT_URL = "http://127.0.0.1:58080"
RISK_TITLE = "FL-M2 risk paragraphs"
RISK_CONFIG = Path("docs/labeling/risk_paragraphs.xml")
RISK_TASKS = Path("data/interim/label_tasks/risk_tasks.json")
EXPORT_DIR = Path("data/interim/label_exports")


def candidates_html(first_seen: dict) -> str:
    """First-seen suggestion and the most similar earlier paragraph per year, as escaped HTML."""
    rows = "".join(
        "<tr>"
        f"<td style='padding:4px;vertical-align:top'><b>{c['year']}</b></td>"
        f"<td style='padding:4px;vertical-align:top'>{c['score']:.2f}</td>"
        f"<td style='padding:4px'><i>{html.escape(c['heading'])}</i><br>{html.escape(c['text'])}"
        " …</td></tr>"
        for c in first_seen["candidates"]
    )
    table = (
        "<table style='border-collapse:collapse;font-size:13px'>"
        "<tr><th align='left'>Year</th><th align='left'>Overlap</th>"
        f"<th align='left'>Most similar paragraph (heading, first words)</th></tr>{rows}</table>"
        if rows
        else "<p>No earlier 10-K for this company in the corpus.</p>"
    )
    return (
        f"<p><b>Suggested first-seen year: {first_seen['suggested']}</b> "
        "(earliest year with word overlap ≥ 0.5; a suggestion only)</p>" + table
    )


def to_ls_task(task: dict) -> dict:
    """One Label Studio task. Split and sampling source are not in the gold tasks file at all."""
    title = (
        f"Task {task['task_no']} · {task['ticker']} · {task['sector']} · "
        f"fiscal {task['period_year']}"
    )
    return {
        "data": {
            "task_no": task["task_no"],
            "para_id": task["para_id"],
            "title": title,
            "heading": task["heading"] or "(no heading)",
            "text": task["text"],
            "period_year": task["period_year"],
            "candidates_html": candidates_html(task["first_seen"]),
        }
    }


class LabelStudio:
    def __init__(self, url: str, token: str, transport: httpx.BaseTransport | None = None):
        self.http = httpx.Client(
            base_url=url,
            headers={"Authorization": f"Token {token}"},
            timeout=120,
            transport=transport,
        )

    def _json(self, method: str, path: str, **kwargs):
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def find_project(self, title: str) -> dict | None:
        body = self._json("GET", "/api/projects", params={"page_size": 1000})
        projects = body["results"] if isinstance(body, dict) else body
        return next((p for p in projects if p["title"] == title), None)

    def create_project(self, title: str, label_config: str) -> dict:
        payload = {"title": title, "label_config": label_config, "maximum_annotations": 1}
        return self._json("POST", "/api/projects", json=payload)

    def task_count(self, project_id: int) -> int:
        return self._json("GET", f"/api/projects/{project_id}")["task_number"]

    def import_tasks(self, project_id: int, tasks: list[dict]) -> dict:
        return self._json("POST", f"/api/projects/{project_id}/import", json=tasks)

    def export(self, project_id: int) -> list[dict]:
        params = {"exportType": "JSON", "download_all_tasks": "true"}
        return self._json("GET", f"/api/projects/{project_id}/export", params=params)


def setup_project(ls: LabelStudio, title: str, config: str, tasks: list[dict]) -> dict:
    """Create the project if missing; import tasks only if it has none."""
    project = ls.find_project(title)
    created = project is None
    if created:
        project = ls.create_project(title, config)
    existing = ls.task_count(project["id"])
    imported = 0
    if existing == 0:
        imported = ls.import_tasks(project["id"], tasks)["task_count"]
    return {
        "project_id": project["id"],
        "created": created,
        "existing": existing,
        "imported": imported,
    }


def export_summary(tasks: list[dict]) -> dict:
    annotated = [t for t in tasks if t.get("annotations")]
    categories = Counter(
        choice
        for t in annotated
        for r in t["annotations"][-1]["result"]
        if r["from_name"] == "category"
        for choice in r["value"]["choices"]
    )
    return {"tasks": len(tasks), "annotated": len(annotated), "categories": dict(categories)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["setup-risk", "export"])
    parser.add_argument("--title", default=RISK_TITLE)
    parser.add_argument("--config", type=Path, default=RISK_CONFIG)
    parser.add_argument("--tasks", type=Path, default=RISK_TASKS)
    parser.add_argument("--out-dir", type=Path, default=EXPORT_DIR)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    token = os.environ.get("LABEL_STUDIO_USER_TOKEN")
    if not token:
        sys.exit("LABEL_STUDIO_USER_TOKEN is not set: run `set -a; source .env; set +a` first")
    ls = LabelStudio(os.environ.get("LABEL_STUDIO_URL", DEFAULT_URL), token)

    if args.command == "setup-risk":
        tasks = [to_ls_task(t) for t in json.loads(args.tasks.read_text())]
        print(json.dumps(setup_project(ls, args.title, args.config.read_text(), tasks), indent=2))
        return

    project = ls.find_project(args.title)
    if project is None:
        sys.exit(f"no project titled {args.title!r}")
    tasks = ls.export(project["id"])
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"risk_paragraphs_{stamp}.json"
    out.write_text(json.dumps(tasks, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"file": str(out), **export_summary(tasks)}, indent=2))


if __name__ == "__main__":
    main()
