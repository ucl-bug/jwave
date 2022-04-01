.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")
JAX_INSTALLED=$(shell python -c "import jax")

.PHONY: clean
clean:            ## Clean unused files.
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '.ipynb_checkpoints' -exec rm -rf {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build

.PHONY: docs
docs:             ## Build the documentation.
	@echo "building documentation ..."
	@$(ENV_PREFIX)pip install -e .[test]
	@$(ENV_PREFIX)mkdocs build
	URL="site/index.html"; xdg-open $$URL || sensible-browser $$URL || x-www-browser $$URL || gnome-open $$URL

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: jaxgpu
jaxgpu:           ## Installs jax for *nix systems with CUDA
	@echo "Installing jax..."
	@$(ENV_PREFIX)pip install --upgrade pip
	@$(ENV_PREFIX)pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html


.PHONY: lint
lint:             ## Runs isort and mypy.
	@echo "Running isort ..."
	$(ENV_PREFIX)isort jwave/
	@echo "Running flake8 ..."
	$(ENV_PREFIX)flake8 jwave/  --count --select=E9,F63,F7,F82 --show-source --statistics
	$(ENV_PREFIX)flake8 jwave/ --count --ignore=E111 --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "Running mypy ..."
	$(ENV_PREFIX)mypy --allow-redefinition --config-file=pyproject.toml jwave/*.py

.PHONY: release
release:          ## Create a new tag for release.
	@echo "WARNING: This operation will create s version tag and push to github"
	@read -p "Version? (provide the next x.y.z semver) : " TAG
	@echo "$${TAG}" > jwave/VERSION
	@$(ENV_PREFIX)gitchangelog > HISTORY.md
	@git add jwave/VERSION HISTORY.md
	@git commit -m "release: version $${TAG} 🚀"
	@echo "creating git tag : $${TAG}"
	@git tag $${TAG}
	@git push -u origin HEAD --tags
	@echo "Github Actions will detect the new tag and release the new version."

.PHONY: serve_docs
serve_docs:       ## Serve the documentation and update it automatically.
	@echo "serving documentation ..."
	@$(ENV_PREFIX)mkdocs serve
	URL="http://localhost:8000/"; xdg-open $$URL || sensible-browser $$URL || x-www-browser $$URL || gnome-open $$URL

.PHONY: show
show:             ## Show the current environment.
	@echo "Current environment:"
	@echo "Running using $(ENV_PREFIX)"
	@$(ENV_PREFIX)python -V
	@$(ENV_PREFIX)python -m site

.PHONY: test
test:             ## Run tests and generate coverage report.
	$(ENV_PREFIX)coverage run --source=jwave -m pytest
	$(ENV_PREFIX)coverage xml
	$(ENV_PREFIX)coverage html

.PHONY: virtualenv
virtualenv:       ## Create a virtual environment. Checks that python > 3.8
	@echo "creating virtual environment ..."
	@python -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8 or higher is required'" || exit 1
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@echo "Instaling JaxDF"
	@./.venv/bin/pip install git+https://github.com/ucl-bug/jaxdf.git
	@echo "Instaling jwave"
	@./.venv/bin/pip install -e .[test]
	@echo "!!! Please run 'source .venv/bin/activate' to enable the environment !!!"
	@echo "--- Don't forget to manually reinstall JAX for GPU/TPU support: https://github.com/google/jax#installation"

.PHONY: watch
watch:            ## Run tests on every change.
	ls **/**.py | entr $(ENV_PREFIX)pytest -s -vvv -l --tb=long --maxfail=1 tests/


# This makefile has been adapted from rochacbruno/python-project-template
