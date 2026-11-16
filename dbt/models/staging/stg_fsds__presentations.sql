-- Where each number appeared: statement, report and line in the filing.
select
    adsh as accession_number,
    report as report_number,
    line as line_number,
    stmt as statement,
    inpth as is_parenthetical,
    rfile as rendered_file,
    tag,
    version as tag_version,
    plabel as presentation_label,
    negating as is_negated,
    fy as fiscal_year,
    source_quarter
from {{ source('fsds', 'pre') }}
