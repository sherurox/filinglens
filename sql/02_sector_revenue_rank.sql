-- 02 Top 5 companies by annual revenue within each sector, fiscal periods ending in 2024.
-- DISTINCT ON picks one revenue tag per company (first in a fixed priority list);
-- RANK() OVER (PARTITION BY sector ...) ranks within the sector, ties share a rank.
with revenue as (
    select distinct on (f.cik)
        f.cik,
        f.value_date,
        f.value as revenue
    from marts.fct_financials as f
    where f.tag in (
            'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax',
            'RevenueFromContractWithCustomerIncludingAssessedTax',
            'SalesRevenueNet'
        )
        and f.duration_quarters = 4
        and f.unit = 'USD'
        and f.value_date >= date '2024-01-01'
        and f.value_date < date '2025-01-01'
    order by
        f.cik,
        array_position(
            array[
                'Revenues',
                'RevenueFromContractWithCustomerExcludingAssessedTax',
                'RevenueFromContractWithCustomerIncludingAssessedTax',
                'SalesRevenueNet'
            ]::text[],
            f.tag::text
        ),
        f.value_date desc
),
ranked as (
    select
        c.sector,
        c.ticker,
        c.company_name,
        r.value_date,
        r.revenue,
        rank() over (partition by c.sector order by r.revenue desc) as sector_rank
    from revenue as r
    inner join marts.dim_company as c on c.cik = r.cik
)
select sector, sector_rank, ticker, company_name, value_date, revenue
from ranked
where sector_rank <= 5
order by sector, sector_rank, ticker;
