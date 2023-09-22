
SELECT DISTINCT P.Postcode, PC.Lat, PC.Long
FROM `bold-circuit-389014.Purchases_Dataset.Purchases_properties_updated` AS P
JOIN `bold-circuit-389014.Postcodes.Postcodes_LSOA_ALL` AS PC
ON P.Postcode = PC.Pcd
ORDER by Postcode