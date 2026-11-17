"""Statement generation for the DuckDB → Postgres copy (the copy itself needs a live Postgres)."""

from filinglens.sql.load_postgres import (
    MART_TABLES,
    copy_statements,
    libpq_dsn,
    postgres_query_sql,
    sql_literal,
)


def test_copy_statements_replace_every_mart_in_the_marts_schema():
    statements = copy_statements()
    assert statements[0] == "CREATE SCHEMA IF NOT EXISTS pg.marts"
    assert len(statements) == 1 + 2 * len(MART_TABLES)
    for table in MART_TABLES:
        drop = f"DROP TABLE IF EXISTS pg.marts.{table}"
        create = f"CREATE TABLE pg.marts.{table} AS SELECT * FROM src.main.{table}"
        assert statements.index(drop) + 1 == statements.index(create)


def test_no_index_or_key_is_created():
    text = " ".join(copy_statements()).upper()
    assert "INDEX" not in text and "PRIMARY KEY" not in text


def test_dsn_is_localhost_on_the_compose_port():
    dsn = libpq_dsn("pw")
    assert dsn == "host=127.0.0.1 port=55432 dbname=filinglens user=filinglens password=pw"


def test_sql_literal_doubles_inner_quotes():
    assert sql_literal("x = 'a'") == "'x = ''a'''"


def test_postgres_query_sql_wraps_the_query_as_one_literal():
    sql = postgres_query_sql("SELECT 1 WHERE n = 'marts'")
    assert sql == "SELECT * FROM postgres_query('pg', 'SELECT 1 WHERE n = ''marts''')"
