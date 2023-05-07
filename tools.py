"""Voice assistant tools."""

from pathlib import Path
from typing import Optional

from deeppavlov import build_model, train_model, Chainer
from deeppavlov.core.common.file import read_json
from pydantic import BaseModel, Field

from config import VOAConfig


class VOAPredictionResult(BaseModel):
    """Voice assistant prediction result."""

    question: str = Field(
        description="Original question"
    )
    answer: str = Field(
        description="Answer to question"
    )
    score: float = Field(
        description="Evaluated score of the answer"
    )
    status: bool = Field(
        description="Status whether the answer to the question is found"
    )

    def __str__(self) -> str:
        return (
            f'Question:\t"{self.question}"\n'
            f'Answer:\t\t"{self.answer}"\n'
            f'Score:\t\t"{self.score}"\n'
            f'Status:\t\t"{self.status}"\n'
        )


class VOATools:
    """Voice assistant tools."""

    def __init__(self, config: VOAConfig):
        """Voice assistant tools."""
        self.config = config

        self.tmp_dir = Path(config.tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True, parents=True)

        self._ml_path = config.ml_config_path
        self._predictor: Optional[Chainer] = None

    def _load_model(self, config_path: str) -> Chainer:
        """Load or train deeppavlov model."""
        model_config = read_json(config_path)
        model_config["metadata"]["variables"]["ROOT_PATH"] = (
            self.config.ml_root_path
        )
        try:
            self._predictor = build_model(model_config, load_trained=True)
        except FileNotFoundError:
            self._predictor = train_model(model_config, download=True)
        return self._predictor

    def load_model(self, config_path: Optional[str] = None) -> Chainer:
        """Return deeppavlov model."""
        if config_path is None or config_path == self._ml_path:
            if self._predictor:
                return self._predictor

        if not config_path:
            config_path = self._ml_path
        predictor = self._load_model(config_path)

        if config_path != self._ml_path:
            self._ml_path = config_path
        return predictor

    def voa_text(
            self,
            question: str,
            default_answer: str = "Извините, не совсем поняла ваш вопрос."
    ) -> VOAPredictionResult:
        """Answer the text question.

        Args:
            question (str, Path): Question.
            default_answer (str): Answer if question is not recognized.

        Returns:
            VOAPredictionResult: result.
        """
        predictor = self.load_model()
        resp = predictor([question])

        answer = resp[0][0]
        score = resp[1][0]
        status = True

        if not score:
            answer = default_answer
            status = False

        return VOAPredictionResult(
            question=question,
            answer=answer,
            score=score,
            status=status
        )
