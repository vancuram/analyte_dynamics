SELECT 
    s.subject_id AS patientid,
    s.stay_id AS stayid,
    s.first_careunit AS careunit,
    s.intime,
    s.outtime,
    s.los,
    CASE 
        WHEN p.gender = 'M' THEN 'male'
        WHEN p.gender = 'F' THEN 'female'
        ELSE 'unknown'
    END AS sex,
    a.age AS age,
    w.weight_admit AS weight,
    h.height,
    CASE
        WHEN p.dod IS NOT NULL 
             AND p.dod >= s.intime 
             AND p.dod <= s.intime + INTERVAL '14 days'
        THEN TRUE
        ELSE FALSE
    END AS death,
    MAX(CASE WHEN d.seq_num = 1 THEN d.name END) AS first_dg,
    MAX(CASE WHEN d.seq_num = 2 THEN d.name END) AS second_dg
FROM mimiciv_icu.icustays AS s
LEFT JOIN mimiciv_hosp.patients AS p 
    ON s.subject_id = p.subject_id 
LEFT JOIN mimiciv_derived.age AS a
    ON s.hadm_id = a.hadm_id
LEFT JOIN mimiciv_derived.first_day_weight AS w
    ON s.stay_id = w.stay_id
LEFT JOIN mimiciv_derived.height AS h
    ON s.stay_id = h.stay_id
LEFT JOIN (
    SELECT 
        dg.hadm_id,
        n.long_title AS name,
        dg.seq_num
    FROM mimiciv_hosp.diagnoses_icd AS dg
    LEFT JOIN mimiciv_hosp.d_icd_diagnoses AS n 
        ON dg.icd_code = n.icd_code
    WHERE dg.seq_num IN (1, 2)
) AS d
    ON s.hadm_id = d.hadm_id
GROUP BY 
    s.subject_id,
    s.stay_id,
    s.first_careunit,
    gender,
    a.age,
    w.weight_admit,
    h.height,
    died_within_2w,
    s.intime,
    s.outtime,
    s.los;

