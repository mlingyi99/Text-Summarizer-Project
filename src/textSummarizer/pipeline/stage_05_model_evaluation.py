from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.model_evaluation import ModelEvaluation
from textSummarizer.logging import logger




class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()