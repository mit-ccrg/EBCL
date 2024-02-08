.PHONY: test
test: ## run tests quickly with the default Python : -k 'test_sample_uniform_regression_preprocess' -s
	python -m pytest --cov=src --cov-report xml 

.PHONY: lint
lint: ## check style with flake8 and isort
	flake8 src tests --max-line-length 88 --extend-ignore E203
	

.PHONY: clean
clean: ## check style with flake8 and isort
	isort --profile black src tests
	black src tests