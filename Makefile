.PHONY: dev list outdated update audit sync

dev:
	uv run uvicorn main:app --reload

list:
	uv run uv pip list

outdated:
	uv run uv pip list --outdated

%:
	@:

update:
	uv run uv add -U $(filter-out $@,$(MAKECMDGOALS))

audit:
	uv run uv audit --preview-features audit-command

freeze:
	uv run uv pip freeze > requirements.txt