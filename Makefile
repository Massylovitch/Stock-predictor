.PHONY: init data baseline train deploy prepare-deployment test-endpoint

DEPLOYMENT_DIR = deployment_dir

init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install
	
data:
	poetry run python -m src.data.make_data

baseline:
	poetry run python -m src.models.model1.train

train:
	poetry run python -m src.models.model2.train 

prepare-deployment:
	rm -rf $(DEPLOYMENT_DIR) && mkdir $(DEPLOYMENT_DIR)
	# poetry export -f requirements.txt --output $(DEPLOYMENT_DIR)/requirements.txt --without-hashes
	cp cerebrium.toml  $(DEPLOYMENT_DIR)/cerebrium.toml
	cp -r src/predict.py $(DEPLOYMENT_DIR)/main.py
	cp -r src $(DEPLOYMENT_DIR)/src/
	
deploy: prepare-deployment
	cd $(DEPLOYMENT_DIR) && poetry run cerebrium deploy --disable-syntax-check