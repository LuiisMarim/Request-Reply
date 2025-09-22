setup:
\tpython -m pip install -r requirements.txt

lint:
\truff check .
\tblack --check .
\tmypy .

format:
\tblack .
\truff check --fix .

test:
\tpytest -q --cov=project

bench:
\tpython -m apps.cli.main benchmark --profile medium --iterations 1000
