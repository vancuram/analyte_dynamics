WITH obs_last_time AS (
  SELECT
    patientid,
    MAX(entertime) AS discharge_time
  FROM `amsterdamumcdb.hirid111.observation`
  GROUP BY patientid
),
first_height AS (
  SELECT patientid, value AS height
  FROM (
    SELECT
      patientid,
      value,
      ROW_NUMBER() OVER (PARTITION BY patientid ORDER BY entertime ASC) AS rn
    FROM `amsterdamumcdb.hirid111.observation`
    WHERE variableid = 10000450
  )
  WHERE rn = 1
),
first_weight AS (
  SELECT patientid, value AS weight
  FROM (
    SELECT
      patientid,
      value,
      ROW_NUMBER() OVER (PARTITION BY patientid ORDER BY entertime ASC) AS rn
    FROM `amsterdamumcdb.hirid111.observation`
    WHERE variableid = 10000400
  )
  WHERE rn = 1
)
SELECT 
    g.patientid AS patientid,
    g.patientid AS stayid,
    g.admissiontime,
    CASE 
        WHEN g.sex = 'M' THEN 'male'
        WHEN g.sex = 'F' THEN 'female'
        ELSE 'unknown'
    END AS sex,
    g.age AS age,
    w.weight,
    h.height,
    CASE
        WHEN g.discharge_status != 'alive' THEN TRUE
        ELSE FALSE
    END AS death,
    DATE_DIFF(o.discharge_time, g.admissiontime, DAY) AS los
FROM `amsterdamumcdb.hirid111.general` AS g
LEFT JOIN obs_last_time o ON g.patientid = o.patientid
LEFT JOIN first_height h ON g.patientid = h.patientid
LEFT JOIN first_weight w ON g.patientid = w.patientid;

--	Observation	10000450	Body height measure	cm
--	Observation	10000400	Body weight	kg	null
