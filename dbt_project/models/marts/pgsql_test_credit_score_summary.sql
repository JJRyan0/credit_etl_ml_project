with base as (
	select * from stg_credit_data_v
)
select 
	job,
	count(*) as Total_applicants,
	avg(is_default::float) as default_rate
from base
group by job
order by default_rate desc;
