-- Table: public.raw_credit_data

-- DROP TABLE public.raw_credit_data;

CREATE TABLE public.raw_credit_data
(
    "unnamed:_0" bigint,
    age bigint,
    sex text COLLATE pg_catalog."default",
    job bigint,
    housing text COLLATE pg_catalog."default",
    saving_accounts text COLLATE pg_catalog."default",
    checking_account text COLLATE pg_catalog."default",
    credit_amount bigint,
    duration bigint,
    purpose text COLLATE pg_catalog."default"
);
