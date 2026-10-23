-- Portfolio schema.
-- Runs once, automatically, the first time the Postgres container starts with an EMPTY data volume
-- (files in /docker-entrypoint-initdb.d run in alphabetical order, as the admin user).

CREATE TABLE users (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email         TEXT        NOT NULL UNIQUE,
    display_name  TEXT        NOT NULL,
    base_currency CHAR(3)     NOT NULL DEFAULT 'USD',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per (user, ticker): the current position.
CREATE TABLE holdings (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id    BIGINT        NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    ticker     TEXT          NOT NULL CHECK (ticker ~ '^[A-Z0-9][A-Z0-9.-]{0,9}$'),
    shares     NUMERIC(20,6) NOT NULL CHECK (shares > 0),
    updated_at TIMESTAMPTZ   NOT NULL DEFAULT now(),
    UNIQUE (user_id, ticker)
);

-- One row per purchase lot of a holding. Cost basis = SUM(shares * price_per_share + fees).
CREATE TABLE cost_basis (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    holding_id      BIGINT        NOT NULL REFERENCES holdings (id) ON DELETE CASCADE,
    purchase_date   DATE          NOT NULL,
    shares          NUMERIC(20,6) NOT NULL CHECK (shares > 0),
    price_per_share NUMERIC(20,6) NOT NULL CHECK (price_per_share >= 0),
    fees            NUMERIC(20,6) NOT NULL DEFAULT 0 CHECK (fees >= 0)
);

CREATE INDEX idx_holdings_user_id     ON holdings (user_id);
CREATE INDEX idx_cost_basis_holding   ON cost_basis (holding_id);
