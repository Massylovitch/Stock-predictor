from typing import Optional
import os

import fire
from cerebrium import deploy, model_type
from comet_ml import API

from src.utils.logger import get_console_logger
from src.utils.paths import MODELS_DIR

logger = get_console_logger(name='model_deployment')

try:
    CEREBRIUM_API_KEY = os.environ['CEREBRIUM_API_KEY']
except KeyError:
    logger.error('CEREBRIUM_API_KEY environment variable not set.')
    raise

def deploy_model(
    from_model_registry: bool = False,
    local_pickle: str = "model.pkl",
):
    
    logger.info('Deploying model...')

    api = API(api_key = os.environ["COMET_ML_API_KEY"],)
    model_pickle_file = str(MODELS_DIR / local_pickle)

    if from_model_registry:
        logger.info('Loading model from model registry...')
        
        # get the Model object
        model = api.get_model(workspace=os.environ["COMET_ML_WORKSPACE"], model_name="Lasso")
        
        # Download a Registry Model:
        model.download("1.0.0", output_folder= MODELS_DIR, expand=True)
    
    elif local_pickle:
        logger.info('Deploying model from local pickle...')
        
    else:
        raise ValueError('Must specify either --local-pickle or --from-model-registry.')

    # https://docs.cerebrium.ai/quickstarts/scikit
    endpoint = deploy((model_type.SKLEARN, model_pickle_file), "sk-test-model" , CEREBRIUM_API_KEY)

    logger.info('Model deployed.')

if __name__ == "__main__":
    fire.Fire(deploy_model)