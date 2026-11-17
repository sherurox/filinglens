-- One row per XBRL tag and taxonomy version (standard or company-specific).
-- Sources are always aliased: in BigQuery an unaliased table name is also a value
-- (the whole row), so `tag` here would mean the row of table `tag`, not the column.
select
    tag,
    version as tag_version,
    custom as is_custom,
    abstract as is_abstract,
    datatype,
    iord as instant_or_duration,
    crdr as credit_or_debit,
    tlabel as tag_label,
    doc as tag_definition,
    source_quarter
from {{ source('fsds', 'tag') }} as src
