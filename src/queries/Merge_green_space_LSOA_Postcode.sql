SELECT P.pcd as Postcode, 
       G.Average_distance_to_nearest_park_or_public_garden__m_, 
       G.Average_number_of_parks_or_public_gardens_within_1_000_m_radius
FROM `bold-circuit-389014.Postcodes.Postcodes_LSOA_ALL` AS P
JOIN `bold-circuit-389014.Green_Space.Green_Space_LSOA` AS G
ON P.lsoa11 = G.LSOA_code
