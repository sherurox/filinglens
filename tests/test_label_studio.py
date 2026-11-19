"""Label Studio client: task building (escaped HTML), idempotent setup and export on a mock API."""

import json

import httpx

from filinglens.extract.label_studio import (
    LabelStudio,
    candidates_html,
    export_summary,
    main,
    setup_project,
    to_ls_task,
)

TASK = {
    "task_no": 7,
    "para_id": "0000320193-25-000079:12",
    "ticker": "AAPL",
    "sector": "manufacturing",
    "period_year": 2024,
    "heading": "",
    "text": "Supply problems could hurt sales.",
    "first_seen": {
        "suggested": 2019,
        "candidates": [
            {"year": 2018, "score": 0.31, "para_id": "a:1", "heading": "Other", "text": "x"},
            {
                "year": 2019,
                "score": 0.8,
                "para_id": "b:2",
                "heading": "<b>Supply</b>",
                "text": "R&D",
            },
        ],
    },
}


class FakeLabelStudio:
    """A tiny in-memory Label Studio API: projects, task counts, import, export."""

    def __init__(self):
        self.projects: dict[int, dict] = {}
        self.tasks: dict[int, list] = {}
        self.auth: set[str] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.auth.add(request.headers.get("authorization", ""))
        path, method = request.url.path, request.method
        if path == "/api/projects" and method == "GET":
            body = {"count": len(self.projects), "results": list(self.projects.values())}
            return httpx.Response(200, json=body)
        if path == "/api/projects" and method == "POST":
            pid = len(self.projects) + 1
            self.projects[pid] = {"id": pid, **json.loads(request.content)}
            self.tasks[pid] = []
            return httpx.Response(201, json=self.projects[pid])
        pid = int(path.split("/")[3])
        if path.endswith("/import"):
            new = json.loads(request.content)
            self.tasks[pid] += new
            return httpx.Response(201, json={"task_count": len(new)})
        if path.endswith("/export"):
            assert request.url.params["download_all_tasks"] == "true"
            return httpx.Response(200, json=self.tasks[pid])
        return httpx.Response(200, json={"id": pid, "task_number": len(self.tasks[pid])})


def test_candidates_html_escapes_filing_text_and_shows_the_suggestion():
    out = candidates_html(TASK["first_seen"])
    assert "Suggested first-seen year: 2019" in out
    assert "&lt;b&gt;Supply&lt;/b&gt;" in out and "R&amp;D" in out and "<b>Supply" not in out
    empty = candidates_html({"suggested": 2019, "candidates": []})
    assert "No earlier 10-K" in empty


def test_to_ls_task_has_title_and_placeholder_heading_but_no_split_or_source():
    data = to_ls_task(TASK)["data"]
    assert data["title"] == "Task 7 · AAPL · manufacturing · fiscal 2024"
    assert data["heading"] == "(no heading)"
    assert "split" not in data and "source" not in data


def test_setup_is_idempotent():
    fake = FakeLabelStudio()
    ls = LabelStudio("http://ls", "tok123", transport=httpx.MockTransport(fake.handler))
    tasks = [to_ls_task(TASK)]
    first = setup_project(ls, "P", "<View/>", tasks)
    second = setup_project(ls, "P", "<View/>", tasks)
    assert (first["created"], first["imported"]) == (True, 1)
    assert (second["created"], second["existing"], second["imported"]) == (False, 1, 0)
    assert len(fake.tasks[1]) == 1 and fake.projects[1]["maximum_annotations"] == 1
    assert fake.auth == {"Token tok123"}


def test_export_summary_counts_latest_annotation_categories():
    tasks = [
        {
            "annotations": [
                {"result": [{"from_name": "category", "value": {"choices": ["people"]}}]}
            ]
        },
        {"annotations": []},
    ]
    assert export_summary(tasks) == {"tasks": 2, "annotated": 1, "categories": {"people": 1}}


def test_main_export_writes_a_timestamped_file(tmp_path, monkeypatch):
    fake = FakeLabelStudio()
    transport = httpx.MockTransport(fake.handler)
    original = LabelStudio.__init__

    def init(self, url, token, transport_=None):
        original(self, url, token, transport=transport)

    monkeypatch.setattr(LabelStudio, "__init__", init)
    monkeypatch.setenv("LABEL_STUDIO_USER_TOKEN", "tok123")
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps([TASK]))
    config = tmp_path / "c.xml"
    config.write_text("<View/>")
    main(["setup-risk", "--tasks", str(tasks_file), "--config", str(config)])
    main(["export", "--out-dir", str(tmp_path / "exports")])
    files = list((tmp_path / "exports").glob("risk_paragraphs_*.json"))
    assert len(files) == 1 and len(json.loads(files[0].read_text())) == 1
