import logging
from pathlib import Path
from typing import Optional

from deeppavlov import Chainer, train_model, build_model
from deeppavlov.core.common.file import read_json
from os.path import join, abspath, dirname
from mycroft import MycroftSkill, intent_file_handler
import sklearn
from mycroft_bus_client import MessageBusClient, Message


class AboutMisis(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self, "AboutMISISSkill")
        self.ml_root_path = './data'

        self.tmp_dir = Path('./tmp_data')
        self.tmp_dir.mkdir(exist_ok=True, parents=True)

        self._ml_path = join(abspath(dirname(__file__)), "data", "tfidf_logreg_autofaq_misis.json")
        self._predictor: Optional[Chainer] = None
        self.is_eye = False

    def initialize(self):
        self.client = MessageBusClient(host='192.168.1.36', port='8000', route='/')


    def handle_check_eye(self, _):
        self.is_eye = True


    @intent_file_handler('misis.about.intent')
    def handle_misis_about(self, message):
        utt = message.data.get('utterance')
        utt = str(utt)
        logging.info("Полученный текст: " + utt)
        if(utt.find("вопрос о мисис") >= 0):
            utt = utt[15:]
        if (utt.find("о мисис") >= 0):
            utt = utt[8:]
        if (utt.find("мисис") >= 0):
            utt = utt[6:]
        if (utt.find("есть вопрос") >= 0):
            utt = utt[12:]
        if (utt.find("хочу задать вопрос") >= 0):
            utt = utt[19:]
        if (utt.find("робот") >= 0):
            utt = utt[6:]
        if (utt.find("answer") >= 0):
            utt = utt[7:]
        if (utt.find("миссис") >= 0):
            utt = utt[7:]
        print('Setting up client to connect to a local mycroft instance')
        logging.info("Отправляем сообщенрие-проверку")
        self.client.run_in_thread()
        self.client.emit(Message('skill.misis.about.eye', data={'check': 'question'}))
        logging.info("вопрос для модели: " + utt)
        self.client.run_in_thread()
        r = self.voa_text(utt)
        logging.info(self.is_eye)
        self.speak(r)


    def _load_model(self, config_path: str) -> Chainer:
        """Load or train deeppavlov model."""
        model_config = read_json(config_path)
        model_config["dataset_reader"]["data_url"] = join(abspath(dirname(__file__)), "data", "dataset_misis_qa.csv")
        model_config["metadata"]["variables"]["ROOT_PATH"] = (
            self.ml_root_path
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
    ) -> str:
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
        logging.info("Полученный ответ: " + answer)
        logging.info("Score: ")
        logging.info(score)
        if not score:
            answer = default_answer
            status = False

        return answer

def create_skill():
    return AboutMisis()

# if __name__ == '__main__':
#     a = AboutMisis()
#     utt = 'направления'
#     r = a.voa_text(utt)
#     print(r)
