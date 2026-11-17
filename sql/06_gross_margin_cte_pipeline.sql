-- 06 Gross margin per company, annual periods ending in 2024, as a CTE pipeline: one revenue
-- and one cost-of-revenue fact per company, joined, computed, then listed by sector.
-- annual_2024 is used twice, so Postgres materializes it once. The year filter is written as
-- extract(year from value_date) = 2024 on purpose: an index on value_date can't serve it
-- (not sargable); an expression index on the same expression can.
with annual_2024 as (
    select cik, tag, value_date, value
    from marts.fct_financials
    where extract(year from value_date) = 2024
        and duration_quarters = 4
        and unit = 'USD'
),
revenue as (
    select distinct on (cik) cik, value_date, value as revenue
    from annual_2024
    where tag in (
            'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax',
            'RevenueFromContractWithCustomerIncludingAssessedTax',
            'SalesRevenueNet'
        )
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
cost as (
    select distinct on (cik) cik, value_date, value as cost_of_revenue
    from annual_2024
    where tag in ('CostOfRevenue', 'CostOfGoodsAndServicesSold')
    order by
        cik,
        array_position(array['CostOfRevenue', 'CostOfGoodsAndServicesSold']::text[], tag::text),
        value_date desc
),
margins as (
    select
        r.cik,
        r.value_date,
        r.revenue,
        c.cost_of_revenue,
        round(100 * (r.revenue - c.cost_of_revenue) / nullif(r.revenue, 0), 2) as gross_margin_pct
    from revenue as r
    inner join cost as c on c.cik = r.cik and c.value_date = r.value_date
)
select d.sector, d.ticker, m.value_date, m.revenue, m.cost_of_revenue, m.gross_margin_pct
from margins as m
inner join marts.dim_company as d on d.cik = m.cik
order by d.sector, m.gross_margin_pct desc, d.ticker;
