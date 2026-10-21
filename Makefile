.PHONY: dev outdated update audit

dev:
	uv run uvicorn main:app --reload

outdated:
	uv run uv pip list --outdated

update:
	uv add -U $(filter-out $@,$(MAKECMDGOALS))

%:
	@:

audit:
	uv audit --preview-features audit-command