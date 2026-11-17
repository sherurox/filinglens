-- 03 Rolling four-quarter average of total assets (the usual return-on-assets denominator),
-- retail sector. Window frame ROWS BETWEEN 3 PRECEDING AND CURRENT ROW; a window counts only
-- when it has 4 balance-sheet dates spanning about nine months (no missing quarter).
-- dim_company is joined once, up front: the planner can't estimate the filter on the window
-- results, and a join after it was planned as a nested loop over every row (T12 finding).
with assets as (
    select c.ticker, f.cik, f.value_date, f.value as assets
    from marts.fct_financials as f
    inner join marts.dim_company as c on c.cik = f.cik
    where f.tag = 'Assets'
        and f.duration_quarters = 0
        and f.unit = 'USD'
        and c.sector = 'retail'
),
rolling as (
    select
        ticker,
        value_date,
        assets,
        avg(assets) over w4 as avg_assets_4q,
        count(*) over w4 as points,
        value_date - min(value_date) over w4 as span_days
    from assets
    window w4 as (partition by cik order by value_date rows between 3 preceding and current row)
)
select ticker, value_date, assets, round(avg_assets_4q, 0) as avg_assets_4q
from rolling
where points = 4 and span_days between 255 and 290
order by ticker, value_date;
