with raw as (
  select * from 
  {{ source('raw', 'raw_credit_data') }}
)
select
	customer_id,
    age,
    job,
    credit_amount,
    duration,
    housing,
    purpose,
	case
		when risk = 'good' then 0 else 1 end as is_default
from raw
