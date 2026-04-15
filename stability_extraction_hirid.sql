WITH admission_times AS (
  SELECT
    patientid AS stayid,
    admissiontime
  FROM `amsterdamumcdb.hirid111.general`
),

labs AS (
  SELECT
    o.patientid AS stayid,
    o.datetime AS charttime,
    CASE o.variableid
      WHEN 20000900 THEN 40762351  -- Hemoglobin
      WHEN 20000400 THEN 3000285   -- Sodium blood
      WHEN 24000605 THEN 3024561   -- Albumin
      WHEN 20002500 THEN 3003458   -- Phosphate
      WHEN 20004300 THEN 3006140   -- Bilirubin
      WHEN 24000230 THEN 3012095   -- Magnesium
      WHEN 24000567 THEN 3032080   -- INR
      WHEN 20004100 THEN 43534077  -- Urea
      WHEN 20000600 THEN 3020564   -- Creatinine
      WHEN 20002200 THEN 3020460   -- CRP
      WHEN 20000700 THEN 3010813   -- Leukocytes
      WHEN 20000110 THEN 3007461   -- Platelets
      WHEN 20002700 THEN 3035995   -- ALP
      WHEN 20005100 THEN 3015377   -- Calcium
    END AS itemid,
    CASE o.variableid
        WHEN 20000900 THEN o.value / 10
        ELSE o.value
    END AS value,
  FROM `amsterdamumcdb.hirid111.observation` o
  WHERE o.variableid IN (
    20000900, 20000400, 24000605, 20002500, 20004300, 24000230, 24000567,
    20004100, 20000600, 20002200, 20000700, 20000110, 20002700, 20005100
  )
),

fluid_agg AS (
  SELECT
    o.patientid AS stayid,
    MAX(o.value) AS fluid_volume_within_24h
  FROM `amsterdamumcdb.hirid111.observation` o
  JOIN admission_times a
    ON o.patientid = a.stayid
   AND o.datetime BETWEEN a.admissiontime AND TIMESTAMP_ADD(a.admissiontime, INTERVAL 24 HOUR)
  WHERE o.variableid = 30005075
  GROUP BY o.patientid
),

pharma_agg AS (
  SELECT
    p.patientid AS stayid,
    SUM(IF(p.pharmaid = 1000100, p.givendose, 0)) AS blood_cells_volume,
    SUM(IF(p.pharmaid = 1000050, p.givendose, 0)) AS plasma_volume,
    SUM(IF(p.pharmaid = 1000201, p.givendose, 0)) AS platelets_volume
  FROM `amsterdamumcdb.hirid111.pharma` p
  JOIN admission_times a
    ON p.patientid = a.stayid
   AND p.givenat BETWEEN a.admissiontime AND TIMESTAMP_ADD(a.admissiontime, INTERVAL 24 HOUR)
  WHERE p.pharmaid IN (1000100, 1000050, 1000201)
  GROUP BY p.patientid
),

max_crp AS (
  SELECT
    o.patientid AS stayid,
    MAX(o.value) AS max_crp
  FROM `amsterdamumcdb.hirid111.observation` o
  JOIN admission_times a
    ON o.patientid = a.stayid
   AND o.datetime BETWEEN a.admissiontime AND TIMESTAMP_ADD(a.admissiontime, INTERVAL 24 HOUR)
  WHERE o.variableid = 20002200
  GROUP BY o.patientid
),

crrt AS (
  SELECT
    o.patientid AS stayid,
    MAX(o.value) > 0 as crrtflag
  FROM `amsterdamumcdb.hirid111.observation` o
  JOIN admission_times a
    ON o.patientid = a.stayid
   AND o.datetime BETWEEN a.admissiontime AND TIMESTAMP_ADD(a.admissiontime, INTERVAL 24 HOUR)
  WHERE o.variableid = 10002508
  GROUP BY o.patientid
)

SELECT
  a.stayid,
  a.admissiontime AS intime,
  l.charttime,
  l.itemid,
  l.value,
  COALESCE(f.fluid_volume_within_24h, 0) AS fluid_volume_within_24h,
  COALESCE(p.blood_cells_volume, 0) AS blood_cells_volume,
  COALESCE(p.plasma_volume, 0) AS plasma_volume,
  COALESCE(p.platelets_volume, 0) AS platelets_volume,
  COALESCE(c.max_crp, -1) AS max_crp,
  cr.crrtflag as crrt,
  0 as albumin_dose_within_24h
FROM admission_times a
LEFT JOIN labs l ON a.stayid = l.stayid
LEFT JOIN pharma_agg p ON a.stayid = p.stayid
LEFT JOIN fluid_agg f ON a.stayid = f.stayid
LEFT JOIN max_crp c ON a.stayid = c.stayid
LEFT JOIN crrt cr ON a.stayid = cr.stayid;




--Hemoglobin [Mass/volume] in Blood	20000900	g/l	223325
--Sodium [Moles/volume] in Blood	20000400	mmol/l	340841
-- Albumin [Mass/volume] in Serum or Plasma	24000605	g/L	13838
-- Phosphate [Moles/volume] in Blood	20002500	mmol/l	132630

--Bilirubin.total [Moles/volume] in Serum or Plasma	20004300	umol/l	78351
-- Magnesium [Moles/volume] in Blood	24000230	mmol/l	130302

--INR in Blood by Coagulation assay	24000567		149453

--Urea [Moles/volume] in Venous blood	20004100	mmol/l	168581


--Creatinine [Moles/volume] in Blood	20000600	umol/l	190527

-- C reactive protein [Mass/volume] in Serum or Plasma	20002200 mg/l 

--Leukocytes [#/volume] in Blood	20000700	G/l	223750

--Platelets [#/volume] in Blood	20000110	G/l	218375
-- Alkaline phosphatase [Enzymatic activity/volume] in Blood	U/l	20002700
--Calcium [Moles/volume] in Blood	20005100	mmol/l	10834
--C reactive protein [Mass/volume] in Serum or Plasma	20002200	mg/l	176654

--Intravenous blood transfusion of packed cells	1000100	ml	43065
--Platelet transfusion	1000201	ml	8415
--Transfusion of plasma (FFP)	1000050	ml	18003	
	


