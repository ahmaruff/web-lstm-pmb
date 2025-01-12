DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS histories;
DROP TABLE IF EXISTS training_logs;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at DATETIME NULL,
    updated_at DATETIME NULL
);

CREATE TABLE histories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    student INTEGER NOT NULL,
    created_at DATETIME NULL,
    updated_at DATETIME NULL
);

CREATE TABLE training_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    loss REAL NULL,
    accuracy REAL,
    training_date DATETIME NULL,
    created_at DATETIME NULL,
    updated_at DATETIME NULL
);
