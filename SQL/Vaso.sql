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

, bp as
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
    , epi.vaso_rate as rate_epinephrine
    , nor.vaso_rate as rate_norepinephrine
    , dop.vaso_rate as rate_dopamine
    , dob.vaso_rate as rate_dobutamine
    , vaso.vaso_rate as rate_vasopressin
    , phenyl.vaso_rate as rate_phenylephrine

  from co

  left join `physionet-data.mimiciii_derived.epinephrine_dose` epi
    on co.icustay_id = epi.icustay_id
    and co.endtime > epi.starttime
    and co.endtime <= epi.endtime
  left join `physionet-data.mimiciii_derived.norepinephrine_dose` nor
    on co.icustay_id = nor.icustay_id
    and co.endtime > nor.starttime
    and co.endtime <= nor.endtime
  left join `physionet-data.mimiciii_derived.dopamine_dose` dop
    on co.icustay_id = dop.icustay_id
    and co.endtime > dop.starttime
    and co.endtime <= dop.endtime
  left join `physionet-data.mimiciii_derived.dobutamine_dose` dob
    on co.icustay_id = dob.icustay_id
    and co.endtime > dob.starttime
    and co.endtime <= dob.endtime
    
  left join `physionet-data.mimiciii_derived.vasopressin_dose` vaso
    on co.icustay_id = vaso.icustay_id
    and co.endtime > vaso.starttime
    and co.endtime <= vaso.endtime
    
   left join `physionet-data.mimiciii_derived.phenylephrine_dose` phenyl
    on co.icustay_id = phenyl.icustay_id
    and co.endtime > phenyl.starttime
    and co.endtime <= phenyl.endtime
  
  
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