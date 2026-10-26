.PHONY: dev list outdated audit freeze install update remove tree

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

install:
	uv run uv pip install -r requirements.txt

update:
	uv run uv add -U $(filter-out $@,$(MAKECMDGOALS))

remove:
	uv run uv remove $(filter-out $@,$(MAKECMDGOALS))

tree:
	uv run uv tree --invert --package $(filter-out $@,$(MAKECMDGOALS))