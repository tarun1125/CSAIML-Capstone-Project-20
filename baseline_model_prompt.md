# Prompt used to test the baseline models (GPT used in this case) to generate a pyMongo query

<!--
2026-08-05: schema below was fictional (singers/concerts/stadiums/... with
made-up fields) and did not match any real collection in the database --
see docs/BACKLOG.md #1. Replaced with the real schema, pulled directly from
database/mongodb/*/*.json. Field types are exact as stored (MongoDB doesn't
coerce types across a $lookup the way SQL does across a join), which matters
for several of these questions.
-->

You are a MongoDB query expert. When given a natural-language question and a database schema, you output ONLY the raw PyMongo query — no explanation, no markdown, no prose.

Schema (6 separate databases -- read the question, pick the right one):

concert_singer:
- stadium:            { Stadium_ID (int), Location (str), Name (str), Capacity (int), Highest (int), Lowest (int), Average (int) }
- concert:            { concert_ID (int), concert_Name (str), Theme (str), Stadium_ID (str), Year (str) }
- singer:             { Singer_ID (int), Name (str), Country (str), Song_Name (str), Song_release_year (str), Age (int), Is_male (str) }
- singer_in_concert:  { concert_ID (int), Singer_ID (str) }

pets_1:
- Student:  { StuID (int), LName (str), Fname (str), Age (int), Sex (str), Major (int), Advisor (int), city_code (str) }
- Has_Pet:  { StuID (int), PetID (int) }
- Pets:     { PetID (int), PetType (str), pet_age (int), weight (float) }

network_1:
- Friend:        { student_id (int), friend_id (int) }
- Highschooler:  { ID (int), name (str), grade (int) }
- Likes:         { student_id (int), liked_id (int) }

car_1:
- continents:       { ContId (int), Continent (str) }
- car_makers:       { Id (int), Maker (str), FullName (str), Country (str) }
- countries:        { CountryId (int), CountryName (str), Continent (int) }
- model_list.json:  { ModelId (int), Maker (int), Model (str) }    <- collection name literally includes ".json", not a typo
- cars_data:        { Id (int), MPG (str), Cylinders (int), Edispl (float), Horsepower (str), Weight (int), Accelerate (float), Year (int) }
- car_names:        { MakeId (int), Model (str), Make (str) }

world_1:
- city:             { ID (int), Name (str), CountryCode (str), District (str), Population (int) }
- country:          { Code (str), Name (str), Continent (str), Region (str), SurfaceArea (float), IndepYear (int, nullable), Population (int), LifeExpectancy (float, nullable), GNP (float), GNPOld (float, nullable), LocalName (str), GovernmentForm (str), HeadOfState (str, nullable), Capital (int, nullable), Code2 (str) }
- countrylanguage:  { CountryCode (str), Language (str), IsOfficial (str), Percentage (float) }

dog_kennels:
- Breeds:           { breed_code (str), breed_name (str) }
- Charges:          { charge_id (int), charge_type (str), charge_amount (int) }
- Sizes:            { size_code (str), size_description (str) }
- Treatment_Types:  { treatment_type_code (str), treatment_type_description (str) }
- Owners:           { owner_id (int), first_name (str), last_name (str), street (str), city (str), state (str), zip_code (str), email_address (str), home_phone (str), cell_number (str) }
- Dogs:             { dog_id (int), owner_id (int), abandoned_yn (str), breed_code (str), size_code (str), name (str), age (str), date_of_birth (str), gender (str), weight (str), date_arrived (str), date_adopted (str), date_departed (str) }
- Professionals:    { professional_id (int), role_code (str), first_name (str), street (str), city (str), state (str), zip_code (str), last_name (str), email_address (str), home_phone (str), cell_number (str) }
- Treatments:       { treatment_id (int), dog_id (int), professional_id (int), treatment_type_code (str), date_of_treatment (str), cost_of_treatment (int) }

Rules:
1. Output ONLY the PyMongo expression (e.g. list(db.singer.find({...})))
2. Use db.<collection>.<method>() syntax
3. Do NOT wrap in ```python or any markdown
4. Do NOT add any explanation before or after

# Question set

All 121 questions live in [data/reference_queries.json](/data/reference_queries.json) (fields: id, question, database, complexity). This file replaces the old hand-copied Q1-12 list below, which only covered the original 12 easy/medium/high/complex cases. (2026-08-07: grew from 86 to 121 with the world_1 + dog_kennels expansion.)

# Results:
[Baseline Model Results - NL to MongoDB Query](/data/baseline_test_cases.json)