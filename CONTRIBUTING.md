# Contributing

Contributions in the form of pull requests are very welcome! Here's how to get started.

<br/>

## Getting started

First, fork the library on GitHub. You can do this by clicking on the `Fork` button in the GitHub interface.

Next, clone and install the library in development mode:

```bash
git clone git@github.com:YOUR_GIT_USERNAME/jwave.git
cd jwave
pip install poetry
poetry install
```

After that, install the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

<br/>

## If you're making changes to the code

Run `git checkout -b my_contribution` and make your changes. Be sure to include additional tests if necessary. Increasing the coverage in the coverage report would be great! 😃

After making your changes, verify that all tests pass.

```bash
coverage run --source=jwave -m pytest -xvs
```

Since regression tests can take a long time to run, it's possible to initially only run unit and integration tests, using:

```bash
coverage run --source=jwave -m pytest -xvs --ignore=tests/regression_tests
```

Once you are satisfied with your changes, add an entry to the changelog using kacl-cli, for example:

```bash
kacl-cli add fixed "Fixed the unfixable issue 🎉" --modify
```

For more information on the types of changes that can be added to the changelog, visit [this page](https://keepachangelog.com/en/1.0.0/).

Then commit and push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub! You can do this by clicking on the Pull Request button in the GitHub interface.

Wait for the CI to run, and one of the developers will review your PR.

<br/>

## If you're making changes to the documentation

Make your changes, and then build the documentation using:

```bash
mkdocs serve
```

Please note that due to the way `operator`s are documented, this might take some time.

You can view your local copy of the documentation by navigating to `localhost:8000` in a web browser.
