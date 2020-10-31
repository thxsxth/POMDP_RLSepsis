-- generate a row for every hour the patient was in the ICU
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
)
,Vaso_MV as(
select icustay_id, itemid,starttime, endtime, -- rate, -- ,rateuom,
case when itemid in (30120,221906,30047) and rateuom='mcg/kg/min' then round(cast(rate as numeric),3)  -- norad
when itemid in (30120,221906,30047) and rateuom='mcg/min' then round(cast(rate/80 as numeric),3)  -- norad
when itemid in (30119,221289) and rateuom='mcg/kg/min' then round(cast(rate as numeric),3) -- epi
when itemid in (30119,221289) and rateuom='mcg/min' then round(cast(rate/80 as numeric),3) -- epi
when itemid in (30051,222315) and rate > 0.2 then round(cast(rate*5/60  as numeric),3) -- vasopressin, in U/h
when itemid in (30051,222315) and rateuom='units/min' then round(cast(rate*5 as numeric),3) -- vasopressin
when itemid in (30051,222315) and rateuom='units/hour' then round(cast(rate*5/60 as numeric),3) -- vasopressin
when itemid in (30128,221749,30127) and rateuom='mcg/kg/min' then round(cast(rate*0.45 as numeric),3) -- phenyl
when itemid in (30128,221749,30127) and rateuom='mcg/min' then round(cast(rate*0.45 / 80 as numeric),3) -- phenyl
when itemid in (221662,30043,30307) and rateuom='mcg/kg/min' then round(cast(rate*0.01 as numeric),3)  -- dopa
when itemid in (221662,30043,30307) and rateuom='mcg/min' then round(cast(rate*0.01/80 as numeric),3) else null end as rate_std-- dopa
from physionet-data.mimiciii_clinical.inputevents_mv
where itemid in (30128,30120,30051,221749,221906,30119,30047,30127,221289,222315,221662,30043,30307) and rate is not null and statusdescription <> 'Rewritten' and icustay_id is not null
order by icustay_id, itemid, starttime),

bp as
(
  select ce.icustay_id
    , ce.charttime
    , min(valuenum) as meanbp_min
  FROM `physionet-data.mimiciii_clinical.chartevents` ce
  -- exclude rows marked as error
  where (ce.error IS NULL OR ce.error != 1)
  and ce.itemid in
  (
  -- MEAN ARTERIAL PRESSURE
  456, --"NBP Mean"
  52, --"Arterial BP Mean"
  6702, --	Arterial BP Mean #2
  443, --	Manual BP Mean(calc)
  220052, --"Arterial Blood Pressure mean"
  220181, --"Non Invasive Blood Pressure mean"
  225312  --"ART BP mean"
  )
  and valuenum > 0 and valuenum < 300
  group by ce.icustay_id, ce.charttime
),

scorecomp as
(
  select
      co.icustay_id
    , co.hr
    , co.starttime, co.endtime    
    , MV.rate_std as vaso_rate
   

  from co

  left join  Vaso_MV MV
    on co.icustay_id = MV.icustay_id
    and co.endtime > MV.starttime
    and co.endtime <= MV.endtime
 
)
, score_final as(
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
