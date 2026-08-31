# Present so `unittest discover -t .` can import the suite with the repo root as
# the top-level directory, which is what makes `from recoup...` resolve without
# an editable install. pytest does not need this, but the zero-install path does.
