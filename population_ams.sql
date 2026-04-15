WITH diagnosis AS (
    SELECT 
        co.visit_occurrence_id AS stayid,
        STRING_AGG(c.concept_name, '& ') AS primary_diagnoses
    FROM `amsterdamumcdb.version1_5_0.condition_occurrence` co
    LEFT JOIN `amsterdamumcdb.version1_5_0.concept` c 
        ON co.condition_concept_id = c.concept_id
    WHERE co.condition_status_concept_id = 32901
    GROUP BY co.visit_occurrence_id
)

SELECT 
    a.admissionid AS stayid,
    a.patientid,
    a.agegroup,
    a.heightgroup,
    a.weightgroup,
    a.specialty,
    ((a.dischargedat - a.admittedat) / (1000*3600*24)) as los, -- Lenght of stay in days
    
    
    -- Gender conversion
    CASE 
        WHEN a.gender = 'Man' THEN 'male'
        WHEN a.gender = 'Vrouw' THEN 'female'
        ELSE 'unknown'
    END AS sex,
    
    -- Check if death occurred within 2 weeks (in ms = 14 * 24 * 60 * 60 * 1000)
    CASE 
        WHEN a.dateofdeath IS NOT NULL 
             AND a.dateofdeath BETWEEN a.admittedat AND (a.admittedat + 1209600000) THEN TRUE
        ELSE FALSE
    END AS died_within_2w,
    
    -- Diagnoses
    diag.primary_diagnoses

FROM `amsterdamumcdb.version1_0_2.admissions` a
LEFT JOIN diagnosis diag 
    ON a.admissionid = diag.stayid
ORDER BY stayid;
