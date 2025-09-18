These files are used as substitution and/or additional files for the rebase tests.

They are meant to be added to output and/or reference folders in various tests,
or to replace files in those folders, in order to simulate things like:
* Old files in the reference folder that need to be removed.
* Changes to filenames due to changes to tests, so that the filename of an output
  file no longer matches the name in the reference folder.
* Mismatches in the contents of reference and output files.

Note that 2 fastq files are just copies of the ones that appear in the test data,
but with there filenames switched. The BAM file matches the name of a file in the
test data, but its contents are a little different. And the summary.txt file is
just one that is completely different from any in the test data.
