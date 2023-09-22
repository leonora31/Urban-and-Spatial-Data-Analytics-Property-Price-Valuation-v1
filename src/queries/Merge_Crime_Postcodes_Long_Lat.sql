CREATE OR REPLACE TABLE `bold-circuit-389014.Crimes_Dataset.Crimes_with_Postcodes` AS
SELECT 
  Crimes.*,
  Postcodes.pcd AS Posctcode,
  Postcodes.lat AS Latitude_Postcode,
  Postcodes.long AS Longitude_Postcode
FROM 
  `bold-circuit-389014.Crimes_Dataset.Crimes` AS Crimes
LEFT JOIN (
  SELECT 
    pcd, 
    lat, 
    long, 
    lsoa11
  FROM (
    SELECT 
      *, 
      ROW_NUMBER() OVER(PARTITION BY lsoa11 ORDER BY pcd) as row_number
    FROM 
      `bold-circuit-389014.Postcodes.Postcodes_LSOA_ALL`
  ) 
  WHERE row_number = 1
) AS Postcodes
ON 
  Crimes.LSOA_code = Postcodes.lsoa11;
