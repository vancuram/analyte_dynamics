SELECT 
    ic.visit_occurrence_id AS stayid,
    ic.visit_start_datetime AS intime,
    ic.visit_end_datetime AS outtime,
    le.measurement_datetime AS charttime,
    le.measurement_concept_id AS itemid,
    CASE
    	WHEN le.measurement_concept_id = 40762351 THEN le.value_as_number * 1.61 -- conversion
    	WHEN le.measurement_concept_id = 42869588 AND le.value_as_number < 1 THEN le.value_as_number * 100 -- conversion hematocrit
    	ELSE le.value_as_number
    END AS value,
    COALESCE(fluids.total_fluid_volume, 0) AS fluid_volume_within_24h,
    COALESCE(albumin.total_albumin_grams, 0) AS albumin_dose_within_24h,
    COALESCE(blood_cells.blood_cells_volume, 0) AS blood_cells_volume,
    COALESCE(plasma.plasma_volume, 0) AS plasma_volume,
    COALESCE(platelets.platelets_volume, 0) AS platelets_volume,
    CASE WHEN crrt.crrt_flag IS NOT NULL THEN TRUE ELSE FALSE END AS crrt,
    COALESCE(crp.max_crp, -1) as max_crp
FROM `amsterdamumcdb.version1_5_0.visit_occurrence` ic
JOIN `amsterdamumcdb.version1_5_0.measurement` le 
    ON ic.visit_occurrence_id = le.visit_occurrence_id 
    AND le.measurement_datetime BETWEEN ic.visit_start_datetime AND ic.visit_end_datetime
    AND le.measurement_concept_id IN (
        3024561, 3003458, 3006140, 3043744, 3005772, 40757494,
        40762351, 3032080, 3015377, 3012095, 43534077, 3000285, 3019550,
        3020564, 3010813, 3007461, 3020460, 42869588, 3035995
    )
    AND le.provider_id IS NOT NULL
    AND (
        le.measurement_concept_id != 40762351
        OR (le.measurement_concept_id = 40762351 AND le.measurement_source_value = 'Hb (bloed)')
    )
    AND (
        le.measurement_concept_id != 42869588
        OR (le.measurement_concept_id = 42869588 AND le.measurement_source_value = 'Ht (bloed)')
    )

-- Fluid volume (normalized)
LEFT JOIN (
    SELECT 
        ie.visit_occurrence_id,
        SUM(
            CASE
                WHEN ie.drug_concept_id = 36249708 THEN ie.quantity / 1
                WHEN ie.drug_concept_id = 40221387 THEN ie.quantity / 9
                WHEN ie.drug_concept_id = 21040129 THEN ie.quantity / 9
                WHEN ie.drug_concept_id = 21118561 THEN ie.quantity / 40
                WHEN ie.drug_concept_id = 40709262 THEN ie.quantity / 1
                WHEN ie.drug_concept_id = 19076324 THEN ie.quantity / 1
                ELSE ie.quantity / 1
            END
        ) AS total_fluid_volume
    FROM `amsterdamumcdb.version1_5_0.drug_exposure` ie
    JOIN `amsterdamumcdb.version1_5_0.visit_occurrence` vo
        ON ie.visit_occurrence_id = vo.visit_occurrence_id
    WHERE ie.drug_concept_id IN (
        36249708, 40221387, 21040129, 21118561, 40709262, 19076324
    )
    AND ie.drug_exposure_start_datetime BETWEEN vo.visit_start_datetime 
                                            AND TIMESTAMP_ADD(vo.visit_start_datetime, INTERVAL 24 HOUR)
    GROUP BY ie.visit_occurrence_id
) AS fluids 
ON ic.visit_occurrence_id = fluids.visit_occurrence_id

-- Albumin
LEFT JOIN (
    SELECT 
        ie.visit_occurrence_id,
        SUM(ie.quantity) / 1000 AS total_albumin_grams
    FROM `amsterdamumcdb.version1_5_0.drug_exposure` ie
    JOIN `amsterdamumcdb.version1_5_0.visit_occurrence` vo
        ON ie.visit_occurrence_id = vo.visit_occurrence_id
    WHERE ie.drug_concept_id = 42482688
    AND ie.drug_exposure_start_datetime BETWEEN vo.visit_start_datetime 
                                           AND TIMESTAMP_ADD(vo.visit_start_datetime, INTERVAL 24 HOUR)
    GROUP BY ie.visit_occurrence_id
) AS albumin
ON ic.visit_occurrence_id = albumin.visit_occurrence_id

-- Blood cells
LEFT JOIN (
    SELECT 
        ie.visit_occurrence_id,
        SUM(ie.quantity) AS blood_cells_volume
    FROM `amsterdamumcdb.version1_5_0.drug_exposure` ie
    JOIN `amsterdamumcdb.version1_5_0.visit_occurrence` vo
        ON ie.visit_occurrence_id = vo.visit_occurrence_id
    WHERE ie.drug_concept_id = 36854116
    AND ie.drug_exposure_start_datetime BETWEEN vo.visit_start_datetime 
                                           AND TIMESTAMP_ADD(vo.visit_start_datetime, INTERVAL 24 HOUR)
    GROUP BY ie.visit_occurrence_id
) AS blood_cells
ON ic.visit_occurrence_id = blood_cells.visit_occurrence_id

-- Plasma
LEFT JOIN (
    SELECT 
        ie.visit_occurrence_id,
        SUM(ie.quantity) AS plasma_volume
    FROM `amsterdamumcdb.version1_5_0.drug_exposure` ie
    JOIN `amsterdamumcdb.version1_5_0.visit_occurrence` vo
        ON ie.visit_occurrence_id = vo.visit_occurrence_id
    WHERE ie.drug_concept_id = 36861661
    AND ie.drug_exposure_start_datetime BETWEEN vo.visit_start_datetime 
                                           AND TIMESTAMP_ADD(vo.visit_start_datetime, INTERVAL 24 HOUR)
    GROUP BY ie.visit_occurrence_id
) AS plasma
ON ic.visit_occurrence_id = plasma.visit_occurrence_id

-- Platelets
LEFT JOIN (
    SELECT 
        ie.visit_occurrence_id,
        SUM(ie.quantity) AS platelets_volume
    FROM `amsterdamumcdb.version1_5_0.drug_exposure` ie
    JOIN `amsterdamumcdb.version1_5_0.visit_occurrence` vo
        ON ie.visit_occurrence_id = vo.visit_occurrence_id
    WHERE ie.drug_concept_id = 36879118
    AND ie.drug_exposure_start_datetime BETWEEN vo.visit_start_datetime 
                                           AND TIMESTAMP_ADD(vo.visit_start_datetime, INTERVAL 24 HOUR)
    GROUP BY ie.visit_occurrence_id
) AS platelets
ON ic.visit_occurrence_id = platelets.visit_occurrence_id

-- CRRT
LEFT JOIN (
    SELECT DISTINCT me.visit_occurrence_id, TRUE AS crrt_flag
    FROM `amsterdamumcdb.version1_5_0.procedure_occurrence` me
    JOIN `amsterdamumcdb.version1_5_0.visit_occurrence` vo
        ON me.visit_occurrence_id = vo.visit_occurrence_id
    WHERE me.procedure_concept_id = 4051330
    AND me.provider_id IS NOT NULL	
    AND me.procedure_datetime BETWEEN vo.visit_start_datetime 
                                   AND TIMESTAMP_ADD(vo.visit_start_datetime, INTERVAL 24 HOUR)
) AS crrt
ON ic.visit_occurrence_id = crrt.visit_occurrence_id

-- Max CRP
LEFT JOIN (
    SELECT 
        me.visit_occurrence_id,
        MAX(me.value_as_number) AS max_crp
    FROM `amsterdamumcdb.version1_5_0.measurement` me
    JOIN `amsterdamumcdb.version1_5_0.visit_occurrence` vo
        ON me.visit_occurrence_id = vo.visit_occurrence_id
    WHERE me.measurement_concept_id = 3020460
    AND me.provider_id IS NOT NULL
    AND me.measurement_datetime BETWEEN vo.visit_start_datetime 
                                  AND TIMESTAMP_ADD(vo.visit_start_datetime, INTERVAL 24 HOUR)
    GROUP BY me.visit_occurrence_id
) AS crp
ON ic.visit_occurrence_id = crp.visit_occurrence_id;
	

-- 3024561 Albumin
-- 3003458 phospate
-- 3006140 Bilirubin.total [Moles/volume] in Serum or Plasma 57312
-- 40762351 Hemoglobin probably from all
-- 3032080 INR in Blood by Coagulation assay
-- 3015377 Calcium [Moles/volume] in Serum or Plasma
-- 3012095 Magnesium [Moles/volume] in Serum or Plasma
-- 43534077 Urea [Moles/volume] in Blood
-- 3000285 Sodium [Moles/volume] in Blood
-- 3019550 Sodium [Moles/volume] in Serum or Plasma
-- 3020564 Creatinine
-- 3010813 Leucocytes/WBC
-- 3007461 Platelets
-- 3020460 C reactive protein [Mass/volume] in Serum or Plasma mg/l
-- 42869588 hematocrit
-- 3035995 Alkaline phosphatase 

-- 3043744 Bilirubin.conjugated+indirect [Mass/volume] in Serum or Plasma 8618
-- 3005772 Bilirubin.conjugated [Moles/volume] in Serum or Plasma 2096
-- 40757494 Bilirubin.total [Moles/volume] in Blood 1720

-- 36249708 500 ML glucose 25 MG/ML / sodium chloride 4.5 MG/ML Injection 133053  conv: 1
-- 40221387 500 ML sodium chloride 9 MG/ML Injection, 76772 conv: 9
-- 21040129 500 ML Sodium Chloride 9 MG/ML Injectable Solution 65470 conv: 9
-- 21118561 500 ML Gelatin 40 MG/ML Injectable Solution [Gelofusine] 47257  conv: 40
-- 40709262 500 ML Calcium Chloride 0.268 MG/ML / Lactate 3.1 MG/ML / Potassium Chloride 0.4 MG/ML / Sodium Chloride 6 MG/ML Injectable Solution [RINGER LACTATE VIAFLO] by Baxter conv: 1
-- 42482688 100 ML Albumin Human, USP 200 MG/ML Injectable Solution conv: 200
-- 19076324 glucose 50 MG/ML Injectable Solution 28568


-- NE 21079483 500 ML Potassium Chloride 75 MG/ML Oral Solution 36032
-- 36854116 HUMAN BLOOD CELLS 21101
-- 36861661 HUMAN PLASMA 5989
-- 36879118 human platelets 3728



-- 4051330 Continuous venovenous hemofiltration

