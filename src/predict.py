import pandas as pd
from comet_ml import API
from sklearn.pipeline import Pipeline
import os
from src.utils.logger import get_console_logger
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import pickle


load_dotenv()
logger = get_console_logger("deployer")

# this code works when running on Cerebrium
app = FastAPI()


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


COMET_ML_WORKSPACE = os.environ["COMET_ML_WORKSPACE"]
COMET_ML_API_KEY = os.environ["COMET_ML_API_KEY"]
COMET_ML_MODEL_NAME = os.environ["COMET_ML_MODEL_NAME"]


def load_production_model_from_registry(
    workspace: str,
    api_key: str,
    model_name: str,
    status: str = "Production",
) -> Pipeline:
    """Loads the production model from the remote model registry"""

    # find model version to deploy
    api = API(api_key)
    model_details = api.get_registry_model_details(workspace, model_name)["versions"]
    model_versions = [md["version"] for md in model_details if md["status"] == status]

    if len(model_versions) == 0:
        logger.error("No production model found")
        raise ValueError("No production model found")
    else:
        logger.info(f"Found {status} model versions: {model_versions}")
        model_version = model_versions[0]

    # download model from comet ml registry to local file
    api.download_registry_model(
        workspace,
        registry_name=model_name,
        version=model_version,
        output_path="./",
        expand=True,
    )

    # load model from local file to memory
    with open("./model.pkl", "rb") as f:
        model = pickle.load(f)

    return model


model = load_production_model_from_registry(
    workspace=COMET_ML_WORKSPACE,
    api_key=COMET_ML_API_KEY,
    model_name=COMET_ML_MODEL_NAME,
)


@app.route("/predict", include_in_schema=False, methods=["GET", "POST"])
async def predict(request):
    body = await request.json()
    instances = body["instances"]

    # transform item to dataframe
    df = pd.DataFrame([instances])

    # predict
    prediction = model.predict(df)[0]

    json_compatible_item_data = jsonable_encoder(prediction)
    return JSONResponse(content=json_compatible_item_data)


if __name__ == "__main__":
    item = {
        "price_24_hour_ago": 46656.851562,
        "price_23_hour_ago": 46700.535156,
        "price_22_hour_ago": 46700.535156,
        "price_21_hour_ago": 46700.535156,
        "price_20_hour_ago": 46700.535156,
        "price_19_hour_ago": 46700.535156,
        "price_18_hour_ago": 46700.535156,
        "price_17_hour_ago": 46700.535156,
        "price_16_hour_ago": 46700.535156,
        "price_15_hour_ago": 46700.535156,
        "price_14_hour_ago": 46700.535156,
        "price_13_hour_ago": 46700.535156,
        "price_12_hour_ago": 46700.535156,
        "price_11_hour_ago": 46700.535156,
        "price_10_hour_ago": 46700.535156,
        "price_9_hour_ago": 46700.535156,
        "price_8_hour_ago": 46700.535156,
        "price_7_hour_ago": 46700.535156,
        "price_6_hour_ago": 46700.535156,
        "price_5_hour_ago": 46700.535156,
        "price_4_hour_ago": 46700.535156,
        "price_3_hour_ago": 46700.535156,
        "price_2_hour_ago": 46700.535156,
        "price_1_hour_ago": 46700.535156,
    }

    # uvicorn.run("src.predict:app", port=5000, log_level="info")

    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/predict/", json={"instances": item})
    assert response.status_code == 200
