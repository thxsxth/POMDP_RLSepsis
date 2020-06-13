-- This table requires:
--  abx_poe_list
--  abx_micro_poe
--  suspinfect_poe

-- DROP TABLE IF EXISTS 'physionet-data.mimiciii_derived.sepsis3cohort';
with co as (
with serv as
(
    select hadm_id, curr_service
    , ROW_NUMBER() over (partition by hadm_id order by transfertime) as rn
    from physionet-data.mimiciii_clinical.services
)
, t1 as
(
select ie.icustay_id, ie.hadm_id
    , ie.intime, ie.outtime
    , DATE_DIFF(DATE(adm.admittime),DATE(pat.dob),YEAR) as age
    , pat.gender
    , adm.ethnicity
    , ie.dbsource
    -- used to get first ICUSTAY_ID
    , ROW_NUMBER() over (partition by ie.subject_id order by intime) as rn

    -- exclusions
    , s.curr_service as first_service
    , adm.HAS_CHARTEVENTS_DATA

    -- suspicion of infection using POE
    , case when spoe.suspected_infection_time is not null then 1 else 0 end
        as suspected_of_infection_poe
    , spoe.suspected_infection_time as suspected_infection_time_poe
    , DATETIME_DIFF(ie.intime,spoe.suspected_infection_time,DAY)
           as suspected_infection_time_poe_days
    , spoe.specimen as specimen_poe
    , spoe.positiveculture as positiveculture_poe
    , spoe.antibiotic_time as antibiotic_time_poe

from physionet-data.mimiciii_clinical.icustays ie
inner join physionet-data.mimiciii_clinical.admissions adm
    on ie.hadm_id = adm.hadm_id
inner join physionet-data.mimiciii_clinical.patients pat
    on ie.subject_id = pat.subject_id
left join serv s
    on ie.hadm_id = s.hadm_id
    and s.rn = 1
left join physionet-data.mimiciii_derived.suspinfect_poe spoe
  on ie.icustay_id = spoe.icustay_id
)
select
    t1.hadm_id, t1.icustay_id
  , t1.intime, t1.outtime

  -- set de-identified ages to median of 91.4
  , case when age > 89 then 91.4 else age end as age
  , gender
  , ethnicity
  , first_service
  , dbsource

  -- suspicion using POE
  , suspected_of_infection_poe
  , suspected_infection_time_poe
  , suspected_infection_time_poe_days
  , specimen_poe
  , positiveculture_poe
  , antibiotic_time_poe

  -- exclusions
  , case when t1.rn = 1 then 0 else 1 end as exclusion_secondarystay
  , case when t1.age <= 16 then 1 else 0 end as exclusion_nonadult
  , case when t1.first_service in ('CSURG','VSURG','TSURG') then 1 else 0 end as exclusion_csurg
  , case when t1.dbsource != 'metavision' then 1 else 0 end as exclusion_carevue
  , case when t1.suspected_infection_time_poe is not null
          and t1.suspected_infection_time_poe < DATETIME_SUB((t1.intime), interval 1 day) then 1
      else 0 end as exclusion_early_suspicion
  , case when t1.suspected_infection_time_poe is not null
          and t1.suspected_infection_time_poe > DATETIME_ADD(t1.intime ,interval 1 day) then 1
      else 0 end as exclusion_late_suspicion
  , case when t1.HAS_CHARTEVENTS_DATA = 0 then 1
         when t1.intime is null then 1
         when t1.outtime is null then 1
      else 0 end as exclusion_bad_data
  -- , case when t1.suspected_of_infection = 0 then 1 else 0 end as exclusion_suspicion

  -- the above flags are used to summarize patients excluded
  -- below flag is used to actually exclude patients in future queries
  , case when
             t1.rn != 1
          or t1.age <= 16
          or t1.first_service in ('CSURG','VSURG','TSURG')
          or t1.HAS_CHARTEVENTS_DATA = 0
          or t1.intime is null
          or t1.outtime is null
          or t1.dbsource != 'metavision'
          or (
                  t1.suspected_infection_time_poe is not null
              and t1.suspected_infection_time_poe < datetime_sub(t1.intime,interval '1' day)
            )
          or (
                  t1.suspected_infection_time_poe is not null
              and t1.suspected_infection_time_poe > datetime_add(t1.intime,interval '1' day)
            )
          -- or t1.suspected_of_infection = 0
            then 1
        else 0 end as excluded
from t1
order by t1.icustay_id)
----------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------

select co.icustay_id, co.hadm_id
    -- exclusion criteria
    , co.excluded
    , co.intime, co.outtime

    , ie.dbsource

    -- suspicion POE
    , co.suspected_infection_time_poe
    , co.suspected_infection_time_poe_days
    , co.specimen_poe
    , co.positiveculture_poe
    , co.antibiotic_time_poe

    -- blood culture on admission
    , bc.charttime as blood_culture_time
    , bc.positiveculture as blood_culture_positive

    , co.age
    , co.gender
    , case when co.gender = 'M' then 1 else 0 end as is_male
    , co.ethnicity

    -- ethnicity flags
    , case when co.ethnicity in
    (
         'WHITE' --  40996
       , 'WHITE - RUSSIAN' --    164
       , 'WHITE - OTHER EUROPEAN' --     81
       , 'WHITE - BRAZILIAN' --     59
       , 'WHITE - EASTERN EUROPEAN' --     25
    ) then 1 else 0 end as race_white
    , case when co.ethnicity in
    (
          'BLACK/AFRICAN AMERICAN' --   5440
        , 'BLACK/CAPE VERDEAN' --    200
        , 'BLACK/HAITIAN' --    101
        , 'BLACK/AFRICAN' --     44
        , 'CARIBBEAN ISLAND' --      9
    ) then 1 else 0 end as race_black
    , case when co.ethnicity in
    (
      'HISPANIC OR LATINO' --   1696
    , 'HISPANIC/LATINO - PUERTO RICAN' --    232
    , 'HISPANIC/LATINO - DOMINICAN' --     78
    , 'HISPANIC/LATINO - GUATEMALAN' --     40
    , 'HISPANIC/LATINO - CUBAN' --     24
    , 'HISPANIC/LATINO - SALVADORAN' --     19
    , 'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)' --     13
    , 'HISPANIC/LATINO - MEXICAN' --     13
    , 'HISPANIC/LATINO - COLOMBIAN' --      9
    , 'HISPANIC/LATINO - HONDURAN' --      4
  ) then 1 else 0 end as race_hispanic
  , case when co.ethnicity not in
  (
      'WHITE' --  40996
    , 'WHITE - RUSSIAN' --    164
    , 'WHITE - OTHER EUROPEAN' --     81
    , 'WHITE - BRAZILIAN' --     59
    , 'WHITE - EASTERN EUROPEAN' --     25
    , 'BLACK/AFRICAN AMERICAN' --   5440
    , 'BLACK/CAPE VERDEAN' --    200
    , 'BLACK/HAITIAN' --    101
    , 'BLACK/AFRICAN' --     44
    , 'CARIBBEAN ISLAND' --      9
    , 'HISPANIC OR LATINO' --   1696
    , 'HISPANIC/LATINO - PUERTO RICAN' --    232
    , 'HISPANIC/LATINO - DOMINICAN' --     78
    , 'HISPANIC/LATINO - GUATEMALAN' --     40
    , 'HISPANIC/LATINO - CUBAN' --     24
    , 'HISPANIC/LATINO - SALVADORAN' --     19
    , 'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)' --     13
    , 'HISPANIC/LATINO - MEXICAN' --     13
    , 'HISPANIC/LATINO - COLOMBIAN' --      9
    , 'HISPANIC/LATINO - HONDURAN' --      4
  ) then 1 else 0 end as race_other
    -- other races
    -- , 'ASIAN' --   1509
    -- , 'ASIAN - CHINESE' --    277
    -- , 'ASIAN - ASIAN INDIAN' --     85
    -- , 'ASIAN - VIETNAMESE' --     53
    -- , 'ASIAN - FILIPINO' --     25
    -- , 'ASIAN - CAMBODIAN' --     17
    -- , 'ASIAN - OTHER' --     17
    -- , 'ASIAN - KOREAN' --     13
    -- , 'ASIAN - JAPANESE' --      7
    -- , 'ASIAN - THAI' --      4
    --
    -- , 'UNKNOWN/NOT SPECIFIED' --   4523
    -- , 'OTHER' --   1512
    -- , 'UNABLE TO OBTAIN' --    814
    -- , 'PATIENT DECLINED TO ANSWER' --    559
    -- , 'MULTI RACE ETHNICITY' --    130
    -- , 'PORTUGUESE' --     61
    -- , 'AMERICAN INDIAN/ALASKA NATIVE' --     51
    -- , 'MIDDLE EASTERN' --     43
    -- , 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER' --     18
    -- , 'SOUTH AMERICAN' --      8
    -- , 'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE' --      3

    

    , ht.Height
    , wt.Weight
    , wt.Weight / (ht.Height/100*ht.Height/100) as bmi

    -- service type on hospital admission
    , co.first_service

    -- outcomes
    , adm.HOSPITAL_EXPIRE_FLAG
    , case when pat.dod <= datetime_add(adm.admittime,interval '30' day) then 1 else 0 end
        as THIRTYDAY_EXPIRE_FLAG
    , ie.los as icu_los
--     , extract(epoch from (adm.dischtime - adm.admittime))/60.0/60.0/24.0 as hosp_los
    , datetime_diff(adm.dischtime,adm.admittime,day) as hosp_los

    -- sepsis flags
    , a.angus as sepsis_angus
    , m.sepsis as sepsis_martin
    , es.sepsis as sepsis_explicit
    , es.septic_shock as septic_shock_explicit
    , es.severe_sepsis as severe_sepsis_explicit
    , nqf.sepsis as sepsis_nqf
    , cdc.sepsis as sepsis_cdc
    , cdc.sepsis_simple as sepsis_cdc_simple

    -- in-hospital mortality score (van Walraven et al.)
--     ,   CONGESTIVE_HEART_FAILURE    *(4)    + CARDIAC_ARRHYTHMIAS   *(4) +
--         VALVULAR_DISEASE            *(-3)   + PULMONARY_CIRCULATION *(0) +
--         PERIPHERAL_VASCULAR         *(0)    + HYPERTENSION*(-1) + PARALYSIS*(0) +
--         OTHER_NEUROLOGICAL          *(7)    + CHRONIC_PULMONARY*(0) +
--         DIABETES_UNCOMPLICATED      *(-1)   + DIABETES_COMPLICATED*(-4) +
--         HYPOTHYROIDISM              *(0)    + RENAL_FAILURE*(3) + LIVER_DISEASE*(4) +
--         PEPTIC_ULCER                *(-9)   + AIDS*(0) + LYMPHOMA*(7) +
--         METASTATIC_CANCER           *(9)    + SOLID_TUMOR*(0) + RHEUMATOID_ARTHRITIS*(0) +
--         COAGULOPATHY                *(3)    + OBESITY*(-5) +
--         WEIGHT_LOSS                 *(4)    + FLUID_ELECTROLYTE         *(6) +
--         BLOOD_LOSS_ANEMIA           *(0)    + DEFICIENCY_ANEMIAS      *(-4) +
--         ALCOHOL_ABUSE               *(0)    + DRUG_ABUSE*(-6) +
--         PSYCHOSES                   *(-5)   + DEPRESSION*(-8)
--     as elixhauser_hospital

    , case when vent.starttime is not null then 1 else 0 end as vent

    , so.sofa as sofa
    , lo.lods as lods
    , si.sirs as sirs
    , qs.qsofa as qsofa

    -- subcomponents for qSOFA
    , qs.SysBP_score as qsofa_sysbp_score
    , qs.GCS_score as qsofa_gcs_score
    , qs.RespRate_score as qsofa_resprate_score

from co
inner join physionet-data.mimiciii_clinical.icustays ie
  on co.icustay_id = ie.icustay_id
inner join physionet-data.mimiciii_clinical.admissions adm
  on ie.hadm_id = adm.hadm_id
inner join physionet-data.mimiciii_clinical.patients pat
  on ie.subject_id = pat.subject_id
-- left join physionet-data.mimiciii_derived.elixhauser_ahrq eli
--   on ie.hadm_id = eli.hadm_id
left join  physionet-data.mimiciii_derived.heightfirstday ht
  on ie.icustay_id = ht.icustay_id
left join  physionet-data.mimiciii_derived.weightfirstday wt
  on ie.icustay_id = wt.icustay_id
left join  physionet-data.mimiciii_derived.angus_sepsis a
  on ie.hadm_id = a.hadm_id
left join  physionet-data.mimiciii_derived.martin_sepsis m
  on ie.hadm_id = m.hadm_id
left join  physionet-data.mimiciii_derived.explicit_sepsis es
  on ie.hadm_id = es.hadm_id
left join  physionet-data.mimiciii_derived.sepsis_nqf_0500 nqf
  on ie.icustay_id = nqf.icustay_id
left join  physionet-data.mimiciii_derived.sepsis_cdc_surveillance cdc
  on ie.icustay_id = cdc.icustay_id
left join  physionet-data.mimiciii_derived.blood_culture_icu_admit bc
  on ie.icustay_id = bc.icustay_id
left join
  ( select icustay_id, min(starttime) as starttime
    from  physionet-data.mimiciii_derived.ventdurations
    group by icustay_id
  ) vent
  on co.icustay_id = vent.icustay_id
  and vent.starttime >= datetime_sub(co.intime,interval '4' hour)
  and vent.starttime <= datetime_sub(co.intime,interval '1' day)
left join  physionet-data.mimiciii_derived.sofa so
  on co.icustay_id = so.icustay_id
left join  physionet-data.mimiciii_derived.sirs si
  on co.icustay_id = si.icustay_id
left join  physionet-data.mimiciii_derived.lods lo
  on co.icustay_id = lo.icustay_id
left join  physionet-data.mimiciii_derived.qsofa qs
  on co.icustay_id = qs.icustay_id
-- left join MLODS ml
--   on co.icustay_id = ml.icustay_id
-- left join QSOFA_admit qsadm
--   on co.icustay_id = qsadm.icustay_id
-- left join SIRS_admit siadm
--   on co.icustay_id = siadm.icustay_id
order by co.icustay_id;

 
