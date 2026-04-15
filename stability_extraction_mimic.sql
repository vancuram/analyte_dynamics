    SELECT
    ic.stay_id as stayid,
    ic.intime,
    ic.outtime,
    le.charttime,
    CASE le.itemid
        WHEN 50862 THEN 3024561
        WHEN 50970 THEN 3003458
        WHEN 50885 THEN 3006140
        WHEN 51222 THEN 40762351
        WHEN 51237 THEN 3032080
        WHEN 50960 THEN 3012095
        WHEN 50893 THEN 3015377
        WHEN 51006 THEN 43534077
        WHEN 50912 THEN 3020564
        WHEN 51301 THEN 3010813
        WHEN 51265 THEN 3007461
        WHEN 50863 THEN 3035995
        WHEN 50983 THEN 3019550
        WHEN 50824 THEN 3000285
        WHEN 50889 THEN 3020460
        WHEN 51221 THEN 42869588
        ELSE NULL
    END AS itemid,
    CASE le.itemid
        WHEN 50970 THEN le.valuenum * 0.323
        WHEN 50862 THEN le.valuenum * 10
        WHEN 50885 THEN le.valuenum * 17.1
        WHEN 51006 THEN le.valuenum * 0.357
        WHEN 50912 THEN le.valuenum * 88.42
        WHEN 50893 THEN le.valuenum * 0.2495
        ELSE le.valuenum
    END AS value,
    COALESCE(fluids.total_fluid_volume, 0) AS fluid_volume_within_24h,
    COALESCE(albumin_25.total_albumin_grams_25 + albumin_5.total_albumin_grams_5, 0) AS albumin_dose_within_24h,
    COALESCE(blood_cells.blood_cells_volume, 0) AS blood_cells_volume,
    COALESCE(plasma.plasma_volume, 0) AS plasma_volume,
    COALESCE(platelets.platelets_volume, 0) AS platelets_volume,
    CASE WHEN crrt.stay_id IS NOT NULL THEN TRUE ELSE FALSE END AS crrt,
    COALESCE(crp.max_crp, -1) as max_crp,
    ic.first_careunit as icu_unit
FROM mimiciv_icu.icustays ic
JOIN mimiciv_hosp.labevents le
    ON ic.hadm_id = le.hadm_id
    AND le.charttime BETWEEN ic.intime AND ic.outtime
    AND le.itemid IN (
        50862, 50970, 50885, 51222, 51237, 50960, 50893, 51006,
        50912, 51301, 51265, 50863, 50983, 50824,50889, 51221
    )
LEFT JOIN (
    SELECT ie.stay_id, SUM(ie.amount) AS total_fluid_volume
    FROM mimiciv_icu.inputevents ie
    JOIN mimiciv_icu.icustays icu ON ie.stay_id = icu.stay_id
    WHERE ie.itemid IN (
        220862, 220864, 225795, 225796,
        220949, 220950, 220952, 225158, 225159, 225161, 228140,
        225825, 225827, 225828, 225944, 226364
    )
    AND ie.starttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    GROUP BY ie.stay_id
) fluids ON ic.stay_id = fluids.stay_id
LEFT JOIN (
    SELECT ie.stay_id, SUM(ie.amount * 0.25) AS total_albumin_grams_25
    FROM mimiciv_icu.inputevents ie
    JOIN mimiciv_icu.icustays icu ON ie.stay_id = icu.stay_id
    WHERE ie.itemid = 220862
    AND ie.starttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    GROUP BY ie.stay_id
) albumin_25 ON ic.stay_id = albumin_25.stay_id
LEFT JOIN (
    SELECT ie.stay_id, SUM(ie.amount * 0.05) AS total_albumin_grams_5
    FROM mimiciv_icu.inputevents ie
    JOIN mimiciv_icu.icustays icu ON ie.stay_id = icu.stay_id
    WHERE ie.itemid = 220864
    AND ie.starttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    GROUP BY ie.stay_id
) albumin_5 ON ic.stay_id = albumin_5.stay_id
LEFT JOIN (
    SELECT ie.stay_id, SUM(ie.amount) AS blood_cells_volume
    FROM mimiciv_icu.inputevents ie
    JOIN mimiciv_icu.icustays icu ON ie.stay_id = icu.stay_id
    WHERE ie.itemid = 225168
    AND ie.starttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    GROUP BY ie.stay_id
) blood_cells ON ic.stay_id = blood_cells.stay_id
LEFT JOIN (
    SELECT ie.stay_id, SUM(ie.amount) AS plasma_volume
    FROM mimiciv_icu.inputevents ie
    JOIN mimiciv_icu.icustays icu ON ie.stay_id = icu.stay_id
    WHERE ie.itemid = 220970
    AND ie.starttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    GROUP BY ie.stay_id
) plasma ON ic.stay_id = plasma.stay_id
LEFT JOIN (
    SELECT ie.stay_id, SUM(ie.amount) AS platelets_volume
    FROM mimiciv_icu.inputevents ie
    JOIN mimiciv_icu.icustays icu ON ie.stay_id = icu.stay_id
    WHERE ie.itemid = 225170
    AND ie.starttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    GROUP BY ie.stay_id
) platelets ON ic.stay_id = platelets.stay_id
LEFT JOIN (
    SELECT DISTINCT c.stay_id
    FROM mimiciv_derived.crrt c
    JOIN mimiciv_icu.icustays icu ON c.stay_id = icu.stay_id
    WHERE c.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
) crrt ON ic.stay_id = crrt.stay_id
LEFT JOIN (
    SELECT le.hadm_id, MAX(le.valuenum) AS max_crp
    FROM mimiciv_hosp.labevents le
    WHERE le.itemid = 50889
    GROUP BY le.hadm_id
) crp ON ic.hadm_id = crp.hadm_id;
--WHERE NOT ic.first_careunit = 'Cardiac Vascular Intensive Care Unit (CVICU)';



-- 220862 | Albumin 25%%,
-- 220864 | Albumin 5%%,
-- 220970 | Fresh Frozen Plasma, 
-- 225168 | Packed Red Blood Cells, 
-- 225170 | Platelets, 
-- 225795 | Dextran 40, 
-- 225796 | Dextran 70
-- 220949 | Dextrose 5%%                   | Dextrose 5%%                   | inputevents | Fluids/Intake | mL       | Solution   |                |                
-- 220950 | Dextrose 10%%                  | Dextrose 10%%                  | inputevents | Fluids/Intake | mL       | Solution   |                |                
-- 220952 | Dextrose 50%%                  | Dextrose 50%%                  | inputevents | Fluids/Intake | mL       | Solution   |                |                
-- 225158 | NaCl 0.9%%                     | NaCl 0.9%%                     | inputevents | Fluids/Intake | mL       | Solution   |                | 
-- 225159 | NaCl 0.45%%                    | NaCl 0.45%%                    | inputevents | Fluids/Intake | mL       | Solution   |                |                
-- 225161 | NaCl 3%% (Hypertonic Saline)   | NaCl 3%% (Hypertonic Saline)   | inputevents | Fluids/Intake | mL 
-- 228140 | Dextrose 20%%                  | Dextrose 20%%                  | inputevents | Fluids/Intake | mL       | Solution   |                |    
-- 225828 | LR 
-- 226364 | OR crystaloid intake
-- 225823 | D5 1/2NS
-- 225825 | D5NS
-- 225944 sterile whater ml
-- 225827 | D5LR  | D5LR         | 

-- 51222 | Hemoglobin              | 4181121
-- 50811 Hemoglobin (Abg)             |  144055
-- 50885 |  Bilirubin, Total             |  1605452
-- 51237 | INR(PT) |   1783315
-- 50862 | Albumin                   |   1032456
-- 50970 | Phosphate                      |  2814319
-- 50960 | Magnesium             |  2933221
-- 50893 | Calcium, Total             |   2969588
-- 51006 | Urea Nitrogen             |   4202807
-- 50912 Kreatinin
-- 51301 White Blood Cells |   K/uL     | 4157283
--  51265 Platelet Count  |   | K/uL     | 4214047
-- 50863 | Alkaline Phosphatase | 1602599
--  50983 | Sodium              |  mEq/L    | 4111288
--  50824 | Sodium, Whole Blood |  mEq/L    |  159073
-- C-Reactive Protein |  50889 | mg/L     | 177983
--  51221 Hematocrit   | percent.       | 4331614






