# Contributing to j-Wave
We would love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix or rethinking a design
- Proposing new features
- Becoming a maintainer

## We Develop with GitHub
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## All Code Changes Happen Through Pull Requests
Pull requests are the best way to propose changes to the codebase (we use [Github Flow](https://guides.github.com/introduction/flow/index.html)). We actively welcome your pull requests:

1. Fork / Clone the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes (and, if you can, increase test coverage 😃).
5. Make sure your code lints.
6. Issue that pull request!

## Report bugs using [Issues](https://github.com/briandk/transcriptase-atom/issues)
We use issues to track public bugs. Report a bug by [opening a new issue](https://bug.medphys.ucl.ac.uk:10080/astanziola/jwave/-/issues/new?issue); it's that easy!

## Write bug reports with detail, background, and sample code
[This is an example](http://stackoverflow.com/q/12488905/180626) of a bug report, and I think it's not a bad model. Here's [another example from Craig Hockenberry](http://www.openradar.me/11905408).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can, such that *anyone* with `jwave` installed can run to reproduce what you see
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports. I'm not even kidding.

## Run tests
Before merging with a main branch or opening a pull request, run the tests and generate the badges via
```bash
# Install dev requirements if not done yet
pip install -r .setup/dev_requirements.txt
# Python syntax errors or undefined names
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# Static type checking
mypy jwave/*.py
# Testing
pytest
```

### References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md)
