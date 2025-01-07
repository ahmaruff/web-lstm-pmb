DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS histories;

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
    year VARCHAR(255) NOT NULL,
    student INTEGER NOT NULL,
    created_at DATETIME NULL,
    updated_at DATETIME NULL
);
