

create view stg_credit_data_v as

select
	customer_id,
    age,
    job,
    credit_amount,
    duration,
    housing,
    purpose,
	CASE
    WHEN credit_amount > 15000 AND duration > 36 THEN 1
    WHEN job IN ('unemployed', 'unskilled') AND housing = 'free' THEN 1
    WHEN purpose = 'radio/tv' AND credit_amount > 12000 THEN 1
    ELSE 0
END AS is_default
	from raw_credit_data;
