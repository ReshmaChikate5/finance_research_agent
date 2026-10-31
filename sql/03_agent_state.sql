-- Checkpoint storage for the agent (Step 15).
-- A SEPARATE database and role, so the agent can write its own state while the portfolio
-- role stays read-only and cannot see agent state.
-- Fresh volume: put this next to 01_schema.sql / 02_roles_and_seed.sql and it runs once, automatically.
-- Existing volume: run it by hand (see the Step 15 walkthrough); it is safe to run once, not twice.
-- The password is a LOCAL DEV default. Never reuse it anywhere real.

CREATE ROLE agent_state_rw LOGIN PASSWORD 'agent_state_local'
    NOSUPERUSER NOCREATEDB NOCREATEROLE;

CREATE DATABASE agent_state OWNER agent_state_rw;

REVOKE ALL ON DATABASE agent_state FROM PUBLIC;
GRANT CONNECT ON DATABASE agent_state TO agent_state_rw;
