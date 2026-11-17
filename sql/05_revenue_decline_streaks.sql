-- 05 Longest streaks of consecutive yearly revenue declines (gaps and islands).
-- One revenue fact per company and period end (tag priority list); a decline needs the prior
-- year present. Among declining years, year - row_number() is constant within a consecutive
-- run, so it labels each island.
with revenue as (
    select distinct on (f.cik, f.value_date)
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
    order by
        f.cik,
        f.value_date,
        array_position(
            array[
                'Revenues',
                'RevenueFromContractWithCustomerExcludingAssessedTax',
                'RevenueFromContractWithCustomerIncludingAssessedTax',
                'SalesRevenueNet'
            ]::text[],
            f.tag::text
        )
),
yearly as (
    select
        cik,
        extract(year from value_date)::int as period_year,
        revenue,
        lag(revenue) over w as prior_revenue,
        lag(extract(year from value_date)::int) over w as prior_year
    from revenue
    window w as (partition by cik order by value_date)
),
declines as (
    select
        cik,
        period_year,
        period_year - row_number() over (partition by cik order by period_year) as island
    from yearly
    where prior_year = period_year - 1 and revenue < prior_revenue
),
streaks as (
    select cik, min(period_year) as first_year, max(period_year) as last_year, count(*) as years
    from declines
    group by cik, island
)
select c.ticker, c.sector, s.first_year, s.last_year, s.years
from streaks as s
inner join marts.dim_company as c on c.cik = s.cik
where s.years >= 3
order by s.years desc, c.ticker, s.first_year;
