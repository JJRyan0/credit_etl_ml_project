with base as (
    select * from {{ ref('stg_credit_data') }}
)

select
    job,
    count(*) as total_applicants,
    avg(is_default::float) as default_rate
from base
group by job
order by default_rate desc
