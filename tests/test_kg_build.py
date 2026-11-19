import csv
import json

from filinglens.extract.entities import Resolver
from filinglens.kg.build import Graph, add_company, analyze, edge_kinds, write_tables
from filinglens.kg.parse import Signer, Subsidiary


def company(cik, name, ticker):
    return {"cik": str(cik), "name": name, "tickers": ticker, "sector": "services"}


CEO = Signer("JANE DOE", "Chief Executive Officer and Director", ("ceo", "director"))
DIRECTOR = Signer("John Q. Smith", "Director", ("director",))
SVP = Signer("Ann Lee", "Senior Vice President", ())
OTHER = Signer("Bob Roe", "Attorney", ())


def test_edge_kinds():
    assert edge_kinds(CEO) == ["has_officer", "has_director"]
    assert edge_kinds(DIRECTOR) == ["has_director"]
    assert edge_kinds(SVP) == ["has_officer"]  # "president" in the title
    assert edge_kinds(OTHER) == ["signed"]


def build_two():
    graph = Graph()
    resolver = Resolver([(2, "Beta Corp"), (99, "Listed Elsewhere Inc")])
    subs = [
        Subsidiary("Alpha Europe B.V.", "Netherlands"),
        Subsidiary("Beta Corporation", "Delaware"),  # exact match to universe company 2
        Subsidiary("Listed Elsewhere, Inc.", "Delaware"),  # listed, but not in the universe
    ]
    add_company(
        graph, company(1, "Alpha Inc", "ALP"), subs, [CEO, DIRECTOR], "a-1", resolver, {1, 2}
    )
    # John Smith also signs for company 2 (a board interlock); "John Smith" = "John Q. Smith"
    add_company(
        graph,
        company(2, "Beta Corp", "BET|BETB"),
        [],
        [Signer("JOHN SMITH", "Director", ("director",))],
        "b-1",
        resolver,
        {1, 2},
    )
    return graph


def test_add_company_nodes_edges_and_resolution():
    graph = build_two()
    assert graph.nodes["company:2"]["attrs"]["ticker"] == "BET"
    assert graph.nodes["subsidiary:1:ALPHA EUROPE BV"]["attrs"]["jurisdiction"] == "Netherlands"
    beta = graph.nodes["subsidiary:1:BETA"]
    assert beta["attrs"]["listed_cik"] == 2
    assert ("subsidiary:1:BETA", "company:2", "same_as") in graph.edges
    assert graph.nodes["subsidiary:1:LISTED ELSEWHERE"]["attrs"]["listed_cik"] == 99
    assert not any(e[1] == "company:99" for e in graph.edges)  # not in the universe: no edge
    assert ("company:1", "person:jane doe", "has_officer") in graph.edges
    assert ("company:1", "person:jane doe", "has_director") in graph.edges
    assert graph.edges[("company:1", "person:john smith", "has_director")]["source"] == "a-1"


def test_analyze_counts_interlocks_and_components():
    stats = analyze(build_two())
    assert stats["nodes_by_kind"] == {"subsidiary": 3, "company": 2, "person": 2}
    assert stats["edges_by_kind"]["has_subsidiary"] == 3
    assert stats["people_on_2plus_companies"] == 1
    assert stats["company_pairs_sharing_a_person"] == 1
    assert stats["connected_components"] == 1
    assert stats["top_interlocked_people"][0] == {
        "id": "person:john smith",
        "label": "John Q. Smith",
        "companies": 2,
    }
    assert stats["top_companies_by_degree"][0]["id"] == "company:1"
    assert stats["subsidiaries_per_company"]["companies_with_none"] == 1


def test_drop_dangling_and_write_tables(tmp_path):
    graph = build_two()
    graph.edge("company:1", "company:404", "same_as", "x")
    assert graph.drop_dangling() == 1
    write_tables(graph, tmp_path)
    with (tmp_path / "edges.csv").open() as f:
        edges = list(csv.DictReader(f))
    assert len(edges) == len(graph.edges)
    assert json.loads(edges[0]["attrs"]) == {"jurisdiction": "Netherlands"}
    with (tmp_path / "nodes.csv").open() as f:
        assert sum(1 for _ in csv.DictReader(f)) == len(graph.nodes)
