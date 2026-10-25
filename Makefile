.PHONY: dev list outdated audit freeze

%:
	@:

dev:
	uv run uvicorn main:app --reload

list:
	uv run uv pip list

outdated:
	uv run uv pip list --outdated

audit:
	uv run uv audit --preview-features audit-command

freeze:
	uv run uv pip freeze > requirements.txt