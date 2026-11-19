"""Knowledge graph: company -> subsidiary (Exhibit 21), company -> officer / director (signatures).

    uv run python -m filinglens.kg.build graph   # parse, build, analyze with NetworkX
    set -a; source .env; set +a                  # POSTGRES_PASSWORD for the next step
    docker compose up -d --wait postgres
    uv run python -m filinglens.kg.build load    # kg.nodes / kg.edges in Postgres, SQL check

Nodes: company:<cik> (the 300 companies), subsidiary:<parent cik>:<name key> (scoped to the parent:
two parents listing "XYZ Capital LLC" are usually different entities), person:<name key>
(lower case, no middle initials or suffixes, so one person signing two 10-Ks is one node;
two different people with the same first and last name would be merged too).
Edges: has_subsidiary, has_officer, has_director, signed (any other title), same_as (a subsidiary
whose name exactly matches a listed company; T13's resolver). Every edge keeps its accession.
"""

import argparse
import csv
import json
import os
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx

from filinglens.extract.entities import Resolver, key, normalize, read_csv, ticker_rows
from filinglens.kg.parse import Signer, parse_ex21, parse_signatures, person_key

OFFICER_WORDS = ("officer", "president", "controller", "treasurer", "secretary", "counsel")
WORK = Path("data/interim/kg")
NODE_FIELDS = ["id", "kind", "label", "attrs"]
EDGE_FIELDS = ["src", "dst", "kind", "attrs", "source"]


def edge_kinds(signer: Signer) -> list[str]:
    kinds = []
    title = signer.title.lower()
    if {"ceo", "cfo", "cao"} & set(signer.roles) or any(w in title for w in OFFICER_WORDS):
        kinds.append("has_officer")
    if "director" in signer.roles:
        kinds.append("has_director")
    return kinds or ["signed"]


class Graph:
    """Nodes and edges as plain dicts, ready for both NetworkX and Postgres."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def node(self, node_id: str, kind: str, label: str, **attrs: Any) -> str:
        if node_id not in self.nodes:
            self.nodes[node_id] = {"id": node_id, "kind": kind, "label": label, "attrs": attrs}
        return node_id

    def edge(self, src: str, dst: str, kind: str, source: str, **attrs: Any) -> None:
        self.edges.setdefault(
            (src, dst, kind),
            {"src": src, "dst": dst, "kind": kind, "attrs": attrs, "source": source},
        )

    def drop_dangling(self) -> int:
        """Remove edges whose endpoint never became a node; return how many."""
        bad = [
            k
            for k, e in self.edges.items()
            if e["src"] not in self.nodes or e["dst"] not in self.nodes
        ]
        for k in bad:
            del self.edges[k]
        return len(bad)

    def to_networkx(self) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for n in self.nodes.values():
            g.add_node(n["id"], kind=n["kind"], label=n["label"])
        for e in self.edges.values():
            g.add_edge(e["src"], e["dst"], key=e["kind"], kind=e["kind"])
        return g


def add_company(
    graph: Graph,
    company: dict[str, str],
    subsidiaries: list,
    signers: list[Signer],
    accession: str,
    resolver: Resolver | None,
    universe: set[int],
) -> None:
    cik = int(company["cik"])
    cid = graph.node(
        f"company:{cik}",
        "company",
        company["name"],
        ticker=company["tickers"].split("|")[0],
        sector=company["sector"],
    )
    for sub in subsidiaries:
        name_key = key(normalize(sub.name))
        if not name_key:
            continue
        resolved = resolver.resolve(sub.name) if resolver else None
        listed = resolved.cik if resolved and resolved.method == "exact" else None
        sid = graph.node(
            f"subsidiary:{cik}:{name_key}",
            "subsidiary",
            sub.name,
            jurisdiction=sub.jurisdiction,
            listed_cik=listed,
        )
        graph.edge(cid, sid, "has_subsidiary", accession, jurisdiction=sub.jurisdiction)
        if listed is not None and listed in universe and listed != cik:
            graph.edge(sid, f"company:{listed}", "same_as", accession, method="exact")
    for signer in signers:
        label = signer.name.title() if signer.name.isupper() else signer.name
        pid = graph.node(f"person:{person_key(signer.name)}", "person", label)
        for kind in edge_kinds(signer):
            graph.edge(cid, pid, kind, accession, title=signer.title, roles=list(signer.roles))


def top(degrees: dict[str, int], graph: Graph, kind: str, n: int = 10) -> list[dict[str, Any]]:
    ranked = sorted(
        ((d, i) for i, d in degrees.items() if graph.nodes[i]["kind"] == kind),
        key=lambda t: (-t[0], t[1]),
    )[:n]
    return [{"id": i, "label": graph.nodes[i]["label"], "degree": d} for d, i in ranked]


def analyze(graph: Graph) -> dict[str, Any]:
    g = graph.to_networkx()
    degree = dict(g.degree())
    undirected = nx.Graph(g)
    components = sorted((len(c) for c in nx.connected_components(undirected)), reverse=True)
    people_companies = {
        n: {u for u, _, k in g.in_edges(n, keys=True) if k != "same_as"}
        for n, d in g.nodes(data=True)
        if d["kind"] == "person"
    }
    interlocked = {p: cs for p, cs in people_companies.items() if len(cs) >= 2}
    company_pairs = Counter(
        tuple(sorted((a, b))) for cs in interlocked.values() for a in cs for b in cs if a < b
    )
    companies = [n for n, d in g.nodes(data=True) if d["kind"] == "company"]
    subs_per = [
        sum(1 for _, _, k in g.out_edges(c, keys=True) if k == "has_subsidiary") for c in companies
    ]
    signers_per = [
        len({v for _, v, k in g.out_edges(c, keys=True) if k != "has_subsidiary"})
        for c in companies
    ]
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "nodes_by_kind": dict(Counter(d["kind"] for _, d in g.nodes(data=True)).most_common()),
        "edges_by_kind": dict(Counter(k for _, _, k in g.edges(keys=True)).most_common()),
        "connected_components": len(components),
        "largest_component_nodes": components[0] if components else 0,
        "companies_in_largest_component": next(
            (
                sum(1 for n in c if g.nodes[n]["kind"] == "company")
                for c in nx.connected_components(undirected)
                if len(c) == components[0]
            ),
            0,
        ),
        "subsidiaries_per_company": {
            "median": statistics.median(subs_per) if subs_per else 0,
            "max": max(subs_per, default=0),
            "companies_with_none": sum(s == 0 for s in subs_per),
        },
        "signers_per_company": {
            "median": statistics.median(signers_per) if signers_per else 0,
            "max": max(signers_per, default=0),
            "companies_with_none": sum(s == 0 for s in signers_per),
        },
        "people_on_2plus_companies": len(interlocked),
        "company_pairs_sharing_a_person": len(company_pairs),
        "top_companies_by_degree": top(degree, graph, "company"),
        "top_people_by_degree": top(degree, graph, "person"),
        "top_interlocked_people": [
            {"id": p, "label": graph.nodes[p]["label"], "companies": len(cs)}
            for p, cs in sorted(interlocked.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]
        ],
    }


def write_tables(graph: Graph, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in (
        ("nodes.csv", graph.nodes.values(), NODE_FIELDS),
        ("edges.csv", graph.edges.values(), EDGE_FIELDS),
    ):
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(r | {"attrs": json.dumps(r["attrs"], ensure_ascii=False)})


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    master = {int(r["cik"]): r for r in read_csv(args.master)}
    tenk = {r["accession"]: r for r in read_csv(args.tenk_manifest)}
    tickers = ticker_rows(json.loads(args.tickers.read_text()))
    resolver = Resolver((r["cik"], r["name"]) for r in tickers)
    graph = Graph()
    coverage = Counter()
    samples = []
    for i, row in enumerate(read_csv(args.ex21_manifest)):
        cik = int(row["cik"])
        if cik not in master:
            continue
        subs = []
        for path in filter(None, row["exhibits"].split("|")):
            data = Path(path).read_bytes()
            coverage["ex21_pdf"] += data[:5] == b"%PDF-"
            subs.extend(parse_ex21(data))
        filing = tenk.get(row["accession"])
        signers = parse_signatures(Path(filing["path"]).read_bytes()) if filing else []
        coverage["companies"] += 1
        coverage["with_subsidiaries"] += bool(subs)
        coverage["with_signers"] += bool(signers)
        coverage["with_ceo"] += any("ceo" in s.roles for s in signers)
        coverage["with_cfo"] += any("cfo" in s.roles for s in signers)
        add_company(graph, master[cik], subs, signers, row["accession"], resolver, set(master))
        if i % 50 == 0:
            samples.append(
                {
                    "ticker": row["ticker"],
                    "subsidiaries": [f"{s.name} | {s.jurisdiction}" for s in subs[:4]],
                    "signers": [f"{s.name} | {s.title} | {','.join(s.roles)}" for s in signers[:5]],
                }
            )
    coverage["dangling_edges_dropped"] = graph.drop_dangling()
    write_tables(graph, args.out)
    stats = {
        "command": "uv run python -m filinglens.kg.build graph",
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "coverage": dict(coverage),
        **analyze(graph),
        "samples": samples,
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    return stats


DDL = """
CREATE SCHEMA IF NOT EXISTS kg;
DROP TABLE IF EXISTS kg.edges;
DROP TABLE IF EXISTS kg.nodes;
CREATE TABLE kg.nodes (
    id text PRIMARY KEY,
    kind text NOT NULL,
    label text NOT NULL,
    attrs jsonb NOT NULL DEFAULT '{}'
);
CREATE TABLE kg.edges (
    src text NOT NULL REFERENCES kg.nodes (id),
    dst text NOT NULL REFERENCES kg.nodes (id),
    kind text NOT NULL,
    attrs jsonb NOT NULL DEFAULT '{}',
    source text NOT NULL,
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX edges_dst_kind ON kg.edges (dst, kind);
"""
INTERLOCKS_SQL = """
SELECT count(*) FROM (
    SELECT e.dst
    FROM kg.edges e
    JOIN kg.nodes n ON n.id = e.dst AND n.kind = 'person'
    WHERE e.kind <> 'same_as'
    GROUP BY e.dst
    HAVING count(DISTINCT e.src) >= 2
) people
"""


def load(args: argparse.Namespace) -> dict[str, Any]:
    import psycopg

    from filinglens.sql.load_postgres import libpq_dsn

    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise SystemExit("set POSTGRES_PASSWORD (set -a; source .env; set +a)")
    started = time.perf_counter()
    with psycopg.connect(libpq_dsn(password, port=args.port)) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        for table, fields in (("nodes", NODE_FIELDS), ("edges", EDGE_FIELDS)):
            with (args.out / f"{table}.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            with cur.copy(f"COPY kg.{table} ({', '.join(fields)}) FROM STDIN") as copy:
                for r in rows:
                    copy.write_row([r[k] for k in fields])
        cur.execute("ANALYZE kg.nodes; ANALYZE kg.edges")
        counts = {}
        for table in ("nodes", "edges"):
            cur.execute(f"SELECT kind, count(*) FROM kg.{table} GROUP BY kind ORDER BY 2 DESC")
            counts[table] = dict(cur.fetchall())
        cur.execute(INTERLOCKS_SQL)
        sql_interlocks = cur.fetchone()[0]
        cur.execute("SHOW server_version")
        version = cur.fetchone()[0]
    graph_stats = json.loads(args.stats.read_text())
    result = {
        "command": "uv run python -m filinglens.kg.build load",
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "postgres_version": version,
        "rows_by_kind": counts,
        "people_on_2plus_companies_sql": sql_interlocks,
        "people_on_2plus_companies_networkx": graph_stats["people_on_2plus_companies"],
        "sql_matches_networkx": sql_interlocks == graph_stats["people_on_2plus_companies"],
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.load_stats.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["graph", "load"])
    parser.add_argument("--master", type=Path, default=Path("gold/company_master.csv"))
    parser.add_argument("--ex21-manifest", type=Path, default=Path("data/raw/ex21/manifest.csv"))
    parser.add_argument("--tenk-manifest", type=Path, default=Path("data/raw/10k/manifest.csv"))
    parser.add_argument("--tickers", type=Path, default=Path("data/raw/company_tickers.json"))
    parser.add_argument("--out", type=Path, default=WORK)
    parser.add_argument("--stats", type=Path, default=Path("bench/kg_stats.json"))
    parser.add_argument("--load-stats", type=Path, default=Path("bench/kg_load.json"))
    parser.add_argument("--port", type=int, default=55432)
    args = parser.parse_args(argv)
    result = build(args) if args.command == "graph" else load(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
