-- This code is derived from https://gitlab.doc.ic.ac.uk/AIClinician/AIClinician/-/blob/master/AIClinician_Data_extract_MIMIC3_140219.ipynb
-- with some minor modfications


WITH co AS
(
  select ih.icustay_id, ie.hadm_id
  , hr
  -- start/endtime can be used to filter to values within this hour
  , DATETIME_SUB(ih.endtime, INTERVAL '1' HOUR) AS starttime
  , ih.endtime
  from `physionet-data.mimiciii_derived.icustay_hours` ih
  INNER JOIN `physionet-data.mimiciii_clinical.icustays` ie
    ON ih.icustay_id = ie.icustay_id
),
fluids_mv as(with t1 as
(
select icustay_id, starttime, endtime, itemid, amount, rate,
case when itemid in (30176,30315) then amount *0.25
when itemid in (30161) then amount *0.3
when itemid in (30020,30015,225823,30321,30186,30211, 30353,42742,42244,225159) then amount *0.5 --
when itemid in (227531) then amount *2.75
when itemid in (30143,225161) then amount *3
when itemid in (30009,220862) then amount *5
when itemid in (30030,220995,227533) then amount *6.66
when itemid in (228341) then amount *8
else amount end as tev -- total equivalent volume
from physionet-data.mimiciii_clinical.inputevents_mv
-- only real time items !!
where icustay_id is not null and amount is not null and itemid in (225158,225943,226089,225168,225828,225823,220862,220970,220864,225159,220995,225170,225825,227533,225161,227531,225171,225827,225941,225823,225825,225941,225825,228341,225827,30018,30021,30015,30296,30020,30066,30001,30030,30060,30005,30321,3000630061,30009,30179,30190,30143,30160,30008,30168,30186,30211,30353,30159,30007,30185,30063,30094,30352,30014,30011,30210,46493,45399,46516,40850,30176,30161,30381,30315,42742,30180,46087,41491,30004,42698,42244)
)


select icustay_id, starttime, endtime, itemid, round(cast(amount as numeric),3) as amount,round(cast(rate as numeric),3) as rate,round(cast(tev as numeric),3) as tev -- total equiv volume
from t1
order by icustay_id, starttime, itemid)

,scorecomp as
(
  select
      co.icustay_id
    , co.hr
    , co.starttime, co.endtime    
    , MV.tev as tev_mv
   

  from co

  left join  fluids_mv MV
    on co.icustay_id = MV.icustay_id
    and co.endtime > MV.starttime
    and co.endtime <= MV.endtime
 
),
score_final as(
select s.*
FROM scorecomp s
WINDOW W as
  (
    PARTITION BY icustay_id
    ORDER BY hr
    ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING
  )
)

select * from score_final
where hr >= 0 
order by icustay_id, hr;
