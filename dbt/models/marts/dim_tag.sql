-- One row per XBRL tag version used by the facts, with its label and definition.
with used as (
    select distinct tag, tag_version
    from {{ ref('int_fsds__company_facts') }}
)

select
    u.tag || '|' || u.tag_version as tag_key,
    u.tag,
    u.tag_version,
    t.tag_label,
    t.tag_definition,
    t.datatype,
    t.is_custom,
    t.is_abstract,
    t.instant_or_duration,
    t.credit_or_debit
from used as u
left join {{ ref('stg_fsds__tags') }} as t
    on u.tag = t.tag and u.tag_version = t.tag_version
