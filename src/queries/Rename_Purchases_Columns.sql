CREATE OR REPLACE TABLE `bold-circuit-389014.Purchases_Dataset.Purchases_properties_updated` AS
SELECT 
    string_field_0 AS `Transaction unique identifier`,
    int64_field_1 AS Price,
    timestamp_field_2 AS `Date of Transfer`,
    string_field_3 AS Postcode,
    string_field_4 AS `Property_Type`,
    bool_field_5 AS `Old_New`,
    string_field_6 AS Duration,
    string_field_7 AS PAON,
    string_field_8 AS SAON,
    string_field_9 AS Street,
    string_field_10 AS Locality,
    string_field_11 AS `Town_City`,
    string_field_12 AS District,
    string_field_13 AS County,
    string_field_14 AS `PPD_Category_Type`,
    string_field_15 AS `Record_Status`
FROM 
    `bold-circuit-389014.Purchases_Dataset.Purchases_properties`
