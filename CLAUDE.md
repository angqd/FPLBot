# fplbot — agent notes

Start with HANDOFF.md (status, next steps, design decisions, gotchas).

**Step 1 is the viability gate in HANDOFF.md:** test the live APIs and check the payloads
before any other work. Getting transfer prices (selling/purchase price, bank, free
transfers from `my-team`) is essential; if you can't get them, it's a blocker to report,
not to work around. Record the results in the handoff's table and get the user's
go-ahead before continuing.

- Install: `pip install -e ".[judge,dev]"`
- Test: `python tests/make_fixture.py && pytest -q` (must stay green)
- Offline run: `fplbot draft --offline tests/fixtures`
- Never set `execute.dry_run: false` or submit transfers without the user's explicit OK.
- Never commit `.env` (holds the FPL refresh token).
- Keep the LLM judge choosing among solver plans only; don't let it generate moves.
- Update HANDOFF.md's Status table when you finish a step.
