{% macro key_hash(columns) -%}
{#- A 64-bit hash of several columns, one per engine. adapter.dispatch picks
    <adapter>__key_hash for the current target, so tests stay engine-neutral. -#}
{{ return(adapter.dispatch('key_hash', 'filinglens')(columns)) }}
{%- endmacro %}

{% macro duckdb__key_hash(columns) -%}
hash({{ columns | join(', ') }})
{%- endmacro %}

{% macro bigquery__key_hash(columns) -%}
{#- BigQuery has no multi-argument hash: serialise the columns (nulls included) to JSON. -#}
farm_fingerprint(to_json_string(struct({{ columns | join(', ') }})))
{%- endmacro %}

{% macro default__key_hash(columns) -%}
{{ exceptions.raise_compiler_error("key_hash has no implementation for " ~ target.type) }}
{%- endmacro %}
