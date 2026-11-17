-- One row per universe company: identity and sector from the seed, details from its latest filing.
with latest_filing as (
    select
        cik,
        company_name,
        fiscal_year_end_mmdd,
        business_state,
        incorporation_state,
        row_number() over (
            partition by cik order by filed_date desc, accepted_at desc
        ) as rn
    from {{ ref('stg_fsds__submissions') }}
    where cik in (select cik from {{ ref('universe') }})
)

select
    u.cik,
    u.ticker,
    coalesce(l.company_name, u.name) as company_name,
    u.sic,
    u.sic_description,
    u.sector,
    l.fiscal_year_end_mmdd,
    l.business_state,
    l.incorporation_state
from {{ ref('universe') }} as u
left join latest_filing as l
    on u.cik = l.cik and l.rn = 1
