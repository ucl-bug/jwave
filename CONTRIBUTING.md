# How to develop on this project

`jwave` welcomes contributions from the community.

## Setting up your own fork of this repo.

- On github interface click on `Fork` button.
- Clone your fork of this repo. `git clone git@github.com:YOUR_GIT_USERNAME/jwave.git`
- Enter the directory `cd jwave`
- Add upstream repo `git remote add upstream https://github.com/ucl-bug/jwave`

## Install the project in develop mode

Run `make install` to install the project in develop mode.

## Run the tests to ensure everything is working

Run `make test` to run the tests.

## Create a new branch to work on your contribution

Run `git checkout -b my_contribution`

## Make your changes

Edit the files using your preferred editor. (we recommend VIM or VSCode)

## Run the linter

Run `make lint` to run the linter.

## Test your changes

Run `make test` to run the tests.

Add tests! It is great if you can increase the coverage in the coverage report 😃.

## Build the docs locally

Run `make docs` to build the docs.

Ensure your new changes are documented.

## Commit your changes

This project uses [conventional git commit messages](https://www.conventionalcommits.org/en/v1.0.0/).

Example: `fix(package): update setup.py arguments 🎉` (emojis are fine too)

## Push your changes to your fork

Run `git push origin my_contribution`

## Submit a pull request

On github interface, click on `Pull Request` button.

Wait CI to run and one of the developers will review your PR.
## Makefile utilities

This project comes with a `Makefile` that contains a number of useful utility.

Type `make help` to see the list of available commands.stanziola.antonio@gmail.com
