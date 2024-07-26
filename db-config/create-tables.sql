CREATE TABLE uwb_results (
    id SERIAL PRIMARY KEY,
    round_id INTEGER NOT NULL,
    anchor_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    distance FLOAT NOT NULL,
    record_time TIMESTAMP NOT NULL
);

CREATE TABLE gnss_results (
    id SERIAL PRIMARY KEY,
    round_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    easting FLOAT NOT NULL,
    northing FLOAT NOT NULL,
    height FLOAT,
    ts TIMESTAMP NOT NULL
);

CREATE TABLE round (
    id SERIAL PRIMARY KEY,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    base_station_anchor INTEGER NOT NULL,
    base_station_northing FLOAT NOT NULL,
    base_station_easting FLOAT NOT NULL,
    metadata TEXT;
);

CREATE TABLE anchor_location (
    id SERIAL PRIMARY KEY,
    anchor_id INTEGER NOT NULL,
    easting FLOAT NOT NULL,
    northing FLOAT NOT NULL,
    height FLOAT
);

CREATE TABLE localised_uwb (
    id SERIAL PRIMARY KEY,
    method_id INTEGER NOT NULL,
    round_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    easting FLOAT NOT NULL,
    northing FLOAT NOT NULL,
    height FLOAT,
    ts TIMESTAMP NOT NULL
);
