import os
from app import app, db

# Delete existing database
db_path = 'pneumonia_app.db'
if os.path.exists(db_path):
    os.remove(db_path)
    print("Deleted existing database")

# Create new database
with app.app_context():
    db.create_all()
    print("Created new database successfully!") 