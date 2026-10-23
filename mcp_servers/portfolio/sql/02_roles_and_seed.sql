-- Read-only role + demo data.
-- Runs after 01_schema.sql, as the admin user, on first container start only.
-- The password below is a LOCAL DEV default. Override via PORTFOLIO_RO_PASSWORD in docker-compose
-- (see the compose snippet) and never reuse it anywhere real.

-- ---- Read-only role for the portfolio MCP server ------------------------------------------
CREATE ROLE portfolio_ro LOGIN PASSWORD 'portfolio_ro_local' NOSUPERUSER NOCREATEDB NOCREATEROLE;

REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT CONNECT ON DATABASE portfolio TO portfolio_ro;
GRANT USAGE   ON SCHEMA public TO portfolio_ro;
GRANT SELECT  ON users, holdings, cost_basis TO portfolio_ro;      -- SELECT only: no INSERT/UPDATE/DELETE
ALTER ROLE portfolio_ro SET default_transaction_read_only = on;    -- belt and braces
ALTER ROLE portfolio_ro SET statement_timeout = '5s';

-- ---- Demo data (fake) ----------------------------------------------------------------------
INSERT INTO users (email, display_name) VALUES
    ('demo@example.com',  'Demo User'),
    ('other@example.com', 'Other User');

INSERT INTO holdings (user_id, ticker, shares) VALUES
    (1, 'NVDA', 50),
    (1, 'AAPL', 30),
    (1, 'MSFT', 20),
    (2, 'AMD',  100);

INSERT INTO cost_basis (holding_id, purchase_date, shares, price_per_share, fees) VALUES
    (1, '2023-03-15', 30, 24.50,  1.00),
    (1, '2024-01-10', 20, 54.20,  1.00),
    (2, '2022-11-02', 30, 145.10, 0.00),
    (3, '2023-06-20', 20, 335.00, 0.00),
    (4, '2023-09-05', 100, 105.75, 2.00);