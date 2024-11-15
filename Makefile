install:
	pip install poetry

initPoetry:
	poetry install

format:
	black *.py

lint:
	pylint --disable=R,C,E1120 *.py


all: test lint format