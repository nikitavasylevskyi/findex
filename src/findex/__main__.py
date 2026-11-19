from findex.cli import app

# The guard matters for the "spawn" start method: worker processes import the
# main module again, and without it every worker would start the CLI.
if __name__ == "__main__":
    app(prog_name="findex")
