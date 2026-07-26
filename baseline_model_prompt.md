# Prompt used to test the baseline models (GPT5 used in this case) to generate a pyMongo query
You are a MongoDB query expert. When given a natural-language question and a database schema, you output ONLY the raw PyMongo query — no explanation, no markdown, no prose. Schema: - singers: { singer_id, name, country, age } - concerts: { concert_id, concert_name, theme, stadium_id, year (string) } - stadiums: { stadium_id, name, capacity, city } - singer_concerts:{ singer_id, concert_id } - pets: { pet_id, pet_type, pet_age, weight } - students: { stu_id, fname, lname, age, sex } - has_pet: { stu_id, pet_id } - cars: { car_id, maker, model, model_year, horsepower, mpg, origin } - highschoolers: { id, name, grade } - likes: { student_id, liked_id } - friends: { student_id, friend_id } Rules: 1. Output ONLY the PyMongo expression (e.g. list(db.singers.find({...}))) 2. Use db.<collection>.<method>() syntax 3. Do NOT wrap in
python or any markdown
4. Do NOT add any explanation before or after

# Easy Complexity
## Query 1:
### How many singers are there?
## Response:
### db.singers.count_documents({})

## Query 2:
### What are the names of singers from France?
## Response:
### list(db.singers.find({"country": "France"}, {"_id": 0, "name": 1}))

## Query 3:
### Find the maximum weight of all pets
## Response:
### list(db.pets.aggregate([
{"$group": {"_id": None, "max_weight": {"$max": "$weight"}}}
]))

# Medium Complexity
## Query 4:
### What are the names and ages of singers, ordered by age?
## Response:
### list(db.singers.find({}, {"_id": 0, "name": 1, "age": 1}).sort("age", 1))

## Query 5:
### How many concerts were held in each stadium?
## Response:
### list(db.concerts.aggregate([
{"$group": {"_id": "$stadium_id", "concert_count": {"$sum": 1}}}
]))

## Query 6:
### Find pet types that have more than one pet, with their counts.
## Response:
### list(db.pets.aggregate([
{"$group": {"_id": "$pet_type", "count": {"$sum": 1}}},
{"$match": {"count": {"$gt": 1}}}
]))

# High Complexity
## Query 7:
### What are the names of singers who performed in a concert in 2014?
## Response:
### list(db.singer_concerts.aggregate([
{
"$lookup": {
"from": "concerts",
"localField": "concert_id",
"foreignField": "concert_id",
"as": "concert"
}
},
{"$unwind": "$concert"},
{"$match": {"concert.year": "2014"}},
{
"$lookup": {
"from": "singers",
"localField": "singer_id",
"foreignField": "singer_id",
"as": "singer"
}
},
{"$unwind": "$singer"},
{
"$project": {
"_id": 0,
"name": "$singer.name"
}
}
]))

## Query 8:
### Find the first names of students who have a dog as a pet.
## Response:
### list(db.has_pet.aggregate([
{
"$lookup": {
"from": "pets",
"localField": "pet_id",
"foreignField": "pet_id",
"as": "pet"
}
},
{"$unwind": "$pet"},
{"$match": {"pet.pet_type": "dog"}},
{
"$lookup": {
"from": "students",
"localField": "stu_id",
"foreignField": "stu_id",
"as": "student"
}
},
{"$unwind": "$student"},
{
"$project": {
"_id": 0,
"fname": "$student.fname"
}
}
]))

## Query 9:
### For each country of origin, find the average miles per gallon of cars.
## Response:
### list(db.cars.aggregate([
{
"$group": {
"_id": "$origin",
"average_mpg": {
"$avg": "$mpg"
}
}
}
]))

# Complex Complexity
## Query 10:
### Find singers who have not participated in any concert.
## Response:
### list(db.singers.aggregate([
{
"$lookup": {
"from": "singer_concerts",
"localField": "singer_id",
"foreignField": "singer_id",
"as": "concerts"
}
},
{
"$match": {
"concerts": {
"$size": 0
}
}
},
{
"$project": {
"_id": 0,
"name": 1
}
}
]))

## Query 11:
### Find students who have more friends than the average number of friends per student.
## Response:
### list(db.friends.aggregate([
{
"$group": {
"_id": "$student_id",
"friend_count": {
"$sum": 1
}
}
},
{
"$group": {
"_id": None,
"avg_friends": {
"$avg": "$friend_count"
},
"students": {
"$push": {
"student_id": "$_id",
"friend_count": "$friend_count"
}
}
}
},
{"$unwind": "$students"},
{
"$match": {
"$expr": {
"$gt": [
"$students.friend_count",
"$avg_friends"
]
}
}
},
{
"$lookup": {
"from": "highschoolers",
"localField": "students.student_id",
"foreignField": "id",
"as": "student"
}
},
{"$unwind": "$student"},
{
"$project": {
"_id": 0,
"name": "$student.name"
}
}
]))

## Query 12:
### For each maker, find the model with the highest horsepower.
## Response:
### list(db.cars.aggregate([
{
"$sort": {
"maker": 1,
"horsepower": -1
}
},
{
"$group": {
"_id": "$maker",
"model": {
"$first": "$model"
},
"horsepower": {
"$first": "$horsepower"
}
}
}
]))