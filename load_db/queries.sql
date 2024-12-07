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