# Engine adapters

Keep each inference backend in its own directory with compatibility notes and smoke tests for the exact packed formats it supports.

Choose the first backend against the reference device and delivery budget. Avoid adding a frontend framework or multiple engines before that experiment. Runtime source code can share one interface while backend dependencies remain local to each adapter.
