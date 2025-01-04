

WITH agg AS (
SELECT h.fx_datetime AS fx_datetime2, count(1) as minutes_count
FROM hours h, minutes m
where m.fx_datetime >= h.fx_datetime
and m.fx_datetime < (h.fx_datetime + interval '1 hour')
group by h.fx_datetime
)
update hours
set minutes = agg.minutes_count
from agg
where fx_datetime = agg.fx_datetime2;


WITH agg AS (
SELECT m.fx_datetime AS fx_datetime2, count(1) as ticks_count
FROM minutes m, ticks t
where t.fx_datetime >= m.fx_datetime
and t.fx_datetime < (m.fx_datetime + interval '1 minute')
group by m.fx_datetime
)
update minutes
set ticks = agg.ticks_count
from agg
where fx_datetime = agg.fx_datetime2;


WITH agg AS (
SELECT h.fx_datetime AS fx_datetime2, sum(m.ticks) as ticks_count
FROM hours h, minutes m
where m.fx_datetime >= h.fx_datetime
and m.fx_datetime < (h.fx_datetime + interval '1 hour')
group by h.fx_datetime
)
update hours
set ticks = agg.ticks_count
from agg
where fx_datetime = agg.fx_datetime2;


WITH agg AS (
SELECT s.id as id2, s.start_time AS fx_datetime2, sum(h.ticks) as ticks_count
FROM weeks s, hours h
where h.fx_datetime >= s.start_time
and h.fx_datetime < s.end_time
group by s.id, s.start_time
)
update weeks
set ticks = agg.ticks_count
from agg
where id = agg.id2;


WITH agg AS (
SELECT s.id as id2, s.start_time AS fx_datetime2, sum(h.minutes) as minutes_count
FROM weeks s, hours h
where h.fx_datetime >= s.start_time
and h.fx_datetime < s.end_time
group by s.id, s.start_time
)
update weeks
set minutes = agg.minutes_count
from agg
where id = agg.id2;

select start_time, end_time, date(start_time)
from sessions
-- where complete is true
where (
   (EXTRACT(MONTH FROM start_time) = 1 and EXTRACT(DAY FROM start_time) = 1) -- new year
or (EXTRACT(MONTH FROM start_time) = 1 and EXTRACT(DAY FROM start_time) = 2) -- day after new year
or (EXTRACT(MONTH FROM start_time) = 1 and EXTRACT(DOW FROM start_time) = 1 and get_week_number(start_time) = 3) -- mlk
or (EXTRACT(MONTH FROM start_time) = 2 and EXTRACT(DOW FROM start_time) = 1 and get_week_number(start_time) = 3) -- washington
or DATE(start_time) in (select friday_date from easters) -- good friday
or DATE(start_time) in (select monday_date from easters) -- easter monday
or (EXTRACT(DOW FROM start_time) = 1 and EXTRACT(MONTH FROM start_time) = 5 and extract(MONTH from (start_time + interval '7 days')) = 6) -- memorial
or (EXTRACT(YEAR FROM start_time) >= 2021 and EXTRACT(DOW FROM start_time) = 5 and EXTRACT(MONTH FROM start_time) = 6 and EXTRACT(DAY FROM start_time) = 18) -- juneteenth
or (EXTRACT(YEAR FROM start_time) >= 2021 and EXTRACT(DOW FROM start_time) > 0 and EXTRACT(DOW FROM start_time) < 6 and EXTRACT(MONTH FROM start_time) = 6 and EXTRACT(DAY FROM start_time) = 19)
or (EXTRACT(YEAR FROM start_time) >= 2021 and EXTRACT(DOW FROM start_time) = 1 and EXTRACT(MONTH FROM start_time) = 6 and EXTRACT(DAY FROM start_time) = 20) -- july 5
or (EXTRACT(DOW FROM start_time) = 5 and EXTRACT(MONTH FROM start_time) = 7 and EXTRACT(DAY FROM start_time) = 3) -- july 4
or (EXTRACT(DOW FROM start_time) > 0 and EXTRACT(DOW FROM start_time) < 6 and EXTRACT(MONTH FROM start_time) = 7 and EXTRACT(DAY FROM start_time) = 4)
or (EXTRACT(DOW FROM start_time) = 1 and EXTRACT(MONTH FROM start_time) = 7 and EXTRACT(DAY FROM start_time) = 5)
or (EXTRACT(MONTH FROM start_time) = 9 and EXTRACT(DOW FROM start_time) = 1 and get_week_number(start_time) = 1) -- labor
or (EXTRACT(MONTH FROM start_time) = 10 and EXTRACT(DOW FROM start_time) = 1 and get_week_number(start_time) = 1) -- columbus
or (EXTRACT(DOW FROM start_time) = 5 and EXTRACT(MONTH FROM start_time) = 11 and EXTRACT(DAY FROM start_time) = 10) -- veterans
or (EXTRACT(DOW FROM start_time) > 0 and EXTRACT(DOW FROM start_time) < 6 and EXTRACT(MONTH FROM start_time) = 11 and EXTRACT(DAY FROM start_time) = 11)
or (EXTRACT(DOW FROM start_time) = 1 and EXTRACT(MONTH FROM start_time) = 11 and EXTRACT(DAY FROM start_time) = 12)
or (EXTRACT(MONTH FROM start_time) = 11 and EXTRACT(DOW FROM start_time) = 4 and get_week_number(start_time) = 4) -- thanksgiving
or (EXTRACT(MONTH FROM start_time) = 11 and EXTRACT(DOW FROM start_time) = 5 and get_week_number(start_time) = 4) -- black friday
or (EXTRACT(MONTH FROM start_time) = 12 and EXTRACT(DAY FROM start_time) = 24)
or (EXTRACT(MONTH FROM start_time) = 12 and EXTRACT(DAY FROM start_time) = 25) -- christmas
or (EXTRACT(MONTH FROM start_time) = 12 and EXTRACT(DAY FROM start_time) = 26)
or (EXTRACT(MONTH FROM start_time) = 12 and EXTRACT(DAY FROM start_time) = 31) -- new years eve
)
order by start_time;


select avg(m.close_price - m.open_price), stddev(m.close_price - m.open_price)
from minutes m, sessions s
where m.fx_datetime >= s.start_time
and m.fx_datetime < s.end_time
and s.complete is true;
-- 0.34379082253957710617	228.498037054401

insert into normals (name, value) values ('candle_average', 0.34379082253957710617);
insert into normals (name, value) values ('candle_stddev', 228.498037054401);

select avg(m.max_price - m.min_price), stddev(m.max_price - m.min_price)
from minutes m, sessions s
where m.fx_datetime >= s.start_time
and m.fx_datetime < s.end_time
and s.complete is true;
-- 304.4657367430532492	247.446640572918

insert into normals (name, value) values ('tail_average', 304.4657367430532492);
insert into normals (name, value) values ('tail_stddev', 247.446640572918);

select avg(m.close_price - m.min_price), stddev(m.close_price - m.min_price)
from minutes m, sessions s
where m.fx_datetime >= s.start_time
and m.fx_datetime < s.end_time
and s.complete is true;
-- 154.3566921288608436	181.362258560728

insert into normals (name, value) values ('above_average', 154.3566921288608436);
insert into normals (name, value) values ('above_stddev', 181.362258560728);

create or replace function get_week_number(date_in timestamp with time zone)
 returns   integer
 language  sql
 immutable strict
as $$
    select cast(FLOOR(EXTRACT(DAY FROM date_in) / 7 - 0.01) as INTEGER) + 1
$$;
