CREATE TABLE `bold-circuit-389014.Crimes_Dataset.Crimes_yearly` AS
SELECT
  EXTRACT(YEAR FROM PARSE_DATE('%Y-%m', Month)) AS Year,
  Latitude,
  Longitude,
  COUNT(*) AS Crime_Count
FROM
  `bold-circuit-389014.Crimes_Dataset.Crimes_total`
WHERE
  Latitude IS NOT NULL
  AND Longitude IS NOT NULL
GROUP BY
  Year,
  Latitude,
  Longitude;