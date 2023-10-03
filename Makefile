.PHONY: init data baseline train deploy prepare-deployment test-endpoint

DEPLOYMENT_DIR = deployment_dir

init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install
	
data:
	poetry run python src/data.py

baseline:
	poetry run python src/baseline_model.py

train:
	poetry run python src/train.py

prepare-deployment:
	rm -rf $(DEPLOYMENT_DIR) && mkdir $(DEPLOYMENT_DIR)
	# poetry export -f requirements.txt --output $(DEPLOYMENT_DIR)/requirements.txt --without-hashes
	cp requirements.txt  $(DEPLOYMENT_DIR)/requirements.txt
	cp -r src/predict.py $(DEPLOYMENT_DIR)/main.py
	cp -r src $(DEPLOYMENT_DIR)/src/
	pip install cerebrium --upgrade # otherwise cerebrium deploy might fail
	
deploy: prepare-deployment
	cd $(DEPLOYMENT_DIR) && poetry run cerebrium deploy --api-key $(CEREBRIUM_API_KEY) --hardware=CPU eth-predictor

test-endpoint:
	poetry run python -m src.test_endpoint