# How To Rebase standard regression tests

## Run The Rebase
Make a commit which starts like "[rebase]*"

## Check The Output

1. The pipeline will run as normal
1. The "regression_test" step(s) will fail
1. The "rebase" step will run
1. This will overwrite the old results with the ones from this pipeline and commit them
1. A new pipeline will be triggered by the rebase commit
1. This pipeline should pass

# How To Rebase When Adding a New Test

In most cases, if a new test has been added, the rebase mechanism will automatically
detect this and add the necessary folders and files to the `ref` folder. The only
condition where the rebase mechanism should not be able to automatically work out
what has happened will be when a platform is removed. The rebase code has no way to
distinguish between a platform having been removed, and something having gone wrong
so that no output was generated for a platform. In such cases it will be necessary
to manually remove the platform `ref` folder.
