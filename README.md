# agentic_store_operations
Exploring agentic AI applications for retail store operations optimization

# agentic_store_operations

# TODO (MANUALLY UPDATE THE SONAR BADGES LINKS) !!!!!!!!!

| Main                                                         | Dev                                                        |
| ------------------------------------------------------------ | ---------------------------------------------------------- |
| [![Coverage][main-coverage]][main-coverage-link]             | [![Coverage][dev-coverage]][dev-coverage-link]             |
| [![Duplicated Lines][main-duplicates]][main-duplicates-link] | [![Duplicated Lines][dev-duplicates]][dev-duplicates-link] |


We follow and feature branch development, OOP paradigm and Test Driven Development.
Please add a new service and test your branch and then create a PR to dev.

## Branching Development

1. Checkout new branch feature/my_feature from dev branch.
2. Commit your changes to your new branch and ensure to add unit tests and have a succesful build.
3. Test your package in Airflow DEV env.
4. If 3 passes merge your branch to DEV. Ensure to always keep your branch up to date from DEV, ie frequently megre DEV into feature/my_feature
5. On the time of a new release, merge DEV to MAIN. Ensure to bump the version in pyproject.toml. We follow [SemVer](https://www.bing.com/search?q=semantic+versioning+computer+s+cience&cvid=277259a75a69464691c2a41fce660be0&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQABhAMgYIAhAAGEAyCAgDEOkHGPxV0gEINTcwNmowajmoAgCwAgE&FORM=ANAB01&PC=U531) and [here](https://semver.org/)
6. Track package in looper [here](https://ci.falcon2.walmart.com/job/sb-segmentations/job/main/) and artifactory in [here](LOOPER_PROJECT)
7. Test package in airflow DEV env and then promote to Airflow Prod ENV by merging airflow git dev branch INTO airflow git main branch.

## Requirements
 Every PR and PUSH in this repo will require your code to be formated and pass all unit tests.

1. Running Pre-commit You can do any of the following two steps:
   1. (Recommended), at the root of your directory please run
      1. `docker pull docker.ci.artifacts.walmart.com/dsi-dataventures-luminate-docker/walmart_dv_ds_horizontal_precommit` ONCE.
      2. `docker tag docker.ci.artifacts.walmart.com/dsi-dataventures-luminate-docker/walmart_dv_ds_horizontal_precommit walmart_dv_ds_horizontal_precommit` ONCE.
      2. Windows: `docker run -it -v ${PWD}:/project --name walmart_dv_ds_horizontal_precommit --rm walmart_dv_ds_horizontal_precommit`
      3. Macbook `docker run -it -v ./:/project --platform linux/amd64 --name walmart_dv_ds_horizontal_precommit --rm walmart_dv_ds_horizontal_precommit`
   2. Install pre-commit as mentioned [here](https://confluence.walmart.com/display/DSIDVL/Pre-Commit+Setup).
      1. If you have pre-commit in your local GLOBALLY, you can run `pre-commit install`. This will enable pre-commit hooks when you a git commit.

2. Running unit tests on code. Please follow Local development setup.
3. In case you do not want to build the package and are JUST 100% updating documenation, please include the word `NO_CI` in your commit.

## Deployment steps.

1. Create a feature branch from DEV.
2. Commit your changes to your branch.
3. Track your pypi package build [here](https://ci.falcon2.walmart.com/job/sb-segmentations/)
4. Create a DAG and Spark Submit file in a feature branch from DEV found [here](https://gecgithub01.walmart.com/dsi-dataventures-luminate/sb-segmentation-dag-pipeline) and update your env.yaml file to include the given python package you created in the above step.


## Local Development Setup

### Docker Build (Recommended)
Use this to use docker as your development enviornment. Note this is equivalent to the docker used in CI/CD in our looper jobs found [here](https://gecgithub01.walmart.com/dsi-dataventures-luminate/spark-docker-images)
#### Building Docker Container
```shell
docker build --tag agentic_store_operations:latest .
```

#### Developing with Docker Container

```shell
docker run -it -v ./:/project --name agentic_store_operations_dev --rm agentic_store_operations:latest
```
Spark UI is NOT accessible. This is for local development now.

with this you can update your code and work as is. If you have Pycharm Enterprise edition
you can add this container as your enviornment.

### No Docker + MacOs

#### Poetry (Without VPN)
1. Install poetry and pyenv. See [here](https://confluence.walmart.com/pages/viewpage.action?spaceKey=DSIDVL&title=MLE+-+Onboarding)
2. Run:
```bash
pyenv install 3.10
pyenv local 3.10
poetry env use 3.10
poetry shell && poetry install
```
3. For pytest run `pytest tests`

#### SDK man (Without VPN)
1. Install SDKman from [here](https://sdkman.io/install)
2. Run for example to install
```bash
sdk list java | grep zulu
sdk install java 11.0.22-zulu
```
3. If Pycharm Test program is not working (ie Java not found) go to Help >> Choose Boot Java Runtime for the IDE >> Use Default Run Time


#### Adding and Removing dependencies
1. Run `poetry add pandas`
2. Run `poetry remove pandas`


[main-coverage]: https://sonar.prod.walmartlabs.com/api/project_badges/measure?branch=main&project=syndicated&metric=coverage&token=sqb_9a13cee458a22e4d47849148b5f034529aaf7d9d
[main-coverage-link]: https://sonar.prod.walmartlabs.com/dashboard?id=syndicated&branch=main
[main-duplicates]: https://sonar.prod.walmartlabs.com/api/project_badges/measure?branch=main&project=syndicated&metric=duplicated_lines_density&token=sqb_9a13cee458a22e4d47849148b5f034529aaf7d9d
[main-duplicates-link]: https://sonar.prod.walmartlabs.com/dashboard?id=syndicated&branch=main


[dev-coverage]: https://sonar.prod.walmartlabs.com/api/project_badges/measure?branch=dev&project=syndicated&metric=coverage&token=sqb_9a13cee458a22e4d47849148b5f034529aaf7d9d
[dev-coverage-link]: https://sonar.prod.walmartlabs.com/dashboard?id=syndicated&branch=dev
[dev-duplicates]: https://sonar.prod.walmartlabs.com/api/project_badges/measure?branch=dev&project=syndicated&metric=duplicated_lines_density&token=sqb_9a13cee458a22e4d47849148b5f034529aaf7d9d
[dev-duplicates-link]: https://sonar.prod.walmartlabs.com/dashboard?id=syndicated&branch=dev
