## Description

**Kindly take a look at [CONTRIBUTING.md](https://github.com/scikit-hep/vector/blob/main/.github/CONTRIBUTING.md).**

_Please describe the purpose of this pull request. Reference and link to any relevant issues or pull requests._

## Checklist

- [ ] Have you followed the guidelines in our Contributing document?
- [ ] Have you checked to ensure there aren't any other open Pull Requests for the required change?
- [ ] Does your submission pass pre-commit? (`$ pre-commit run --all-files` or `$ nox -s lint`)
- [ ] Does your submission pass tests? (`$ pytest` or `$ nox -s tests`)
- [ ] Does the documentation build with your changes? (`$ cd docs; make clean; make html` or `$ nox -s docs`)
- [ ] Does your submission pass the doctests? (`$ pytest --doctest-plus src/vector/` or `$ nox -s doctests`)

## Before Merging

- [ ] Summarize the commit messages into a brief review of the Pull request.
