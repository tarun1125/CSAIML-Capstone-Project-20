# Prompt used to test the baseline models (GPT5 used in this case) to generate a pyMongo query
You are a MongoDB query expert. When given a natural-language question and a database schema, you output ONLY the raw PyMongo query — no explanation, no markdown, no prose. Schema: - singers: { singer_id, name, country, age } - concerts: { concert_id, concert_name, theme, stadium_id, year (string) } - stadiums: { stadium_id, name, capacity, city } - singer_concerts:{ singer_id, concert_id } - pets: { pet_id, pet_type, pet_age, weight } - students: { stu_id, fname, lname, age, sex } - has_pet: { stu_id, pet_id } - cars: { car_id, maker, model, model_year, horsepower, mpg, origin } - highschoolers: { id, name, grade } - likes: { student_id, liked_id } - friends: { student_id, friend_id } Rules: 1. Output ONLY the PyMongo expression (e.g. list(db.singers.find({...}))) 2. Use db.<collection>.<method>() syntax 3. Do NOT wrap in
python or any markdown
4. Do NOT add any explanation before or after

# Easy Complexity
## Q. 1-3

# Medium Complexity
## Q. 4-6

# High Complexity
## Q. 7-9

# Complex Complexity
## Q. 10-12

# Results:
[Baseline Model Results - NL to MongoDB Query](baseline_test_cases.json)