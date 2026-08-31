# Convenience wrapper around run.py, which is the real entry point.
# Windows users without GNU make should call run.py directly:
#     python run.py demo
#
# There are no `api` or `web` targets. Both existed here before either layer did,
# which is a worse failure than their absence: a Makefile target that does not run
# is a promise the repository makes and breaks. See "What is not here" in the README.

.PHONY: install install-core test lint demo compare sensitivity model verify clean

install:
	python -m pip install -e ".[model,dev]"

# The domain core needs no dependencies; this target exists to prove it.
install-core:
	@echo "Nothing to install. Run: make test"

test:
	python -m unittest discover -s tests -t . -q

lint:
	ruff check recoup tests
	ruff format --check recoup tests

demo:
	python run.py demo

compare:
	python run.py compare

sensitivity:
	python run.py sensitivity

# Needs numpy: make install
model:
	python run.py model --reliability

verify:
	python run.py verify --tamper

clean:
	rm -rf .pytest_cache .ruff_cache artifacts data/*.jsonl
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
