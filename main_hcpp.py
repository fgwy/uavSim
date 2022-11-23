from dataclasses import dataclass

from src.gym.h_cpp import h_CPPGymParams, h_CPPGym
from src.trainer.ddqn import DDQNTrainerParams, DDQNTrainer
from src.base.evaluator import Evaluator, EvaluatorParams
from src.base.logger import LoggerParams, Logger
from utils import AbstractParams


@dataclass
class h_CPPParams(AbstractParams):
    trainer: DDQNTrainerParams = DDQNTrainerParams()
    gym: h_CPPGymParams = h_CPPGymParams()
    logger: LoggerParams = LoggerParams()
    evaluator: EvaluatorParams = EvaluatorParams()


if __name__ == "__main__":

    params, args = h_CPPParams.from_args()
    log_dir = params.create_folders(args)

    logger = Logger(params.logger, log_dir)
    gym = h_CPPGym(params.gym)
    trainer = DDQNTrainer(params.trainer, gym, logger)
    evaluator = Evaluator(params.evaluator, trainer, gym)
    logger.evaluator = evaluator

    trainer.train()
