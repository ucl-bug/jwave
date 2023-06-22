# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

<br/>

## Getting started

First fork the library on GitHub. On github interface click on `Fork` button.

Then clone and install the library in development mode:

```bash
git clone git@github.com:YOUR_GIT_USERNAME/jwave.git
cd jwave
pip install poetry
poetry install
```

Then install the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

<br/>

## If you're making changes to the code

Run `git checkout -b my_contribution` and make your changes. Make sure to include additional tests if necessary. It is great if you can increase the coverage in the coverage report 😃.

Next verify the tests all pass.

```bash
coverage run --source=jwave -m pytest -xvs
```

Since regression tests can take a long time to run, it is possible to initially only run unit and integration tests, using

```bash
coverage run --source=jwave -m pytest -xvs --ignore=tests/regression_tests
```

Once you are happy with your changes, you can add an entry in the changelog using `kacl-cli`, for example

```bash
kacl-cli add fixed "Fixed the unfixable issue 🎉" --modify
```

For more informations on the kind of changes that can be added to the changelog, see [this page](https://keepachangelog.com/en/1.0.0/).

Then commit and push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub! On github interface, click on `Pull Request` button.

Wait CI to run and one of the developers will review your PR.

<br/>

## If you're making changes to the documentation

Make your changes. You can then build the documentation by doing

```bash
mkdocs serve
```

Note that this is going to take some time due to the way `operator`s are documented, please be patient.

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.
