-- 08 Net margin quartiles per sector (PERCENTILE_CONT ... WITHIN GROUP), annual periods
-- ending in 2024. One revenue (tag priority list) and one NetIncomeLoss fact per company.
with facts as (
    select cik, tag, value_date, value
    from marts.fct_financials
    where tag in (
            'NetIncomeLoss',
            'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax',
            'RevenueFromContractWithCustomerIncludingAssessedTax',
            'SalesRevenueNet'
        )
        and duration_quarters = 4
        and unit = 'USD'
        and value_date >= date '2024-01-01'
        and value_date < date '2025-01-01'
),
revenue as (
    select distinct on (cik) cik, value_date, value as revenue
    from facts
    where tag <> 'NetIncomeLoss'
    order by
        cik,
        array_position(
            array[
                'Revenues',
                'RevenueFromContractWithCustomerExcludingAssessedTax',
                'RevenueFromContractWithCustomerIncludingAssessedTax',
                'SalesRevenueNet'
            ]::text[],
            tag::text
        ),
        value_date desc
),
net_income as (
    select distinct on (cik) cik, value_date, value as net_income
    from facts
    where tag = 'NetIncomeLoss'
    order by cik, value_date desc
),
margins as (
    select r.cik, n.net_income / r.revenue as net_margin
    from revenue as r
    inner join net_income as n on n.cik = r.cik and n.value_date = r.value_date
    where r.revenue > 0
)
select
    c.sector,
    count(*) as companies,
    round((percentile_cont(0.25) within group (order by m.net_margin))::numeric * 100, 2) as p25_pct,
    round((percentile_cont(0.50) within group (order by m.net_margin))::numeric * 100, 2) as median_pct,
    round((percentile_cont(0.75) within group (order by m.net_margin))::numeric * 100, 2) as p75_pct
from margins as m
inner join marts.dim_company as c on c.cik = m.cik
group by c.sector
order by c.sector;
