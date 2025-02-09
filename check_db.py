import sqlite3
from app import app, db

# Connect to the database
conn = sqlite3.connect('pneumonia_app.db')
cursor = conn.cursor()

# Show all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("\nTables in the database:")
for table in tables:
    print(f"- {table[0]}")

# Show structure of users table
cursor.execute("PRAGMA table_info(users);")
columns = cursor.fetchall()
print("\nStructure of users table:")
for col in columns:
    print(f"- {col[1]} ({col[2]})")

# Show all users (if any)
cursor.execute("SELECT id, name, email, created_at FROM users;")
users = cursor.fetchall()
print("\nRegistered users:")
for user in users:
    print(f"- ID: {user[0]}, Name: {user[1]}, Email: {user[2]}, Created: {user[3]}")

conn.close()

with app.app_context():
    db.create_all() 