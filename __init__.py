import json
import logging
import socket
from os.path import join, abspath, dirname
from pathlib import Path
from typing import Optional
from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov import configs, train_model
import openai
from deeppavlov import Chainer, train_model, build_model
from deeppavlov.core.common.file import read_json
from mycroft import MycroftSkill, intent_file_handler, intent_handler
from mycroft.skills.intent_service import AdaptIntent


class AboutMisis(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self, "AboutMISISSkill")
        self.ml_root_path = './data'

        self.tmp_dir = Path('./tmp_data')
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        openai.api_key = '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'
        self._ml_path = join(abspath(dirname(__file__)), "data", "tfidf_logreg_autofaq_misis.json")
        self._predictor: Optional[Chainer] = None
        self.is_eye = False
        self.TCP_IP = '192.168.1.101'
        self.TCP_PORT = 5005
        self.BUFFER_SIZE = 128
        self.MESSAGE = json.dumps({'type': 'eye'})
        self.to_person = 'Пожалуйста, вернитесь в фокус зрения робота'
        self.error_message = 'Есть технический сбой в моих системах произвожу перезагрузку'

    @intent_file_handler('misis.about.intent')
    # @intent_handler(AdaptIntent()
    #                 .one_of("университет", "мисис", "вопрос"))
    def handle_misis_about(self, message):
        try:
            utt = message.data.get('utterance')
            utt = str(utt)
            logging.info("Полученный текст: " + utt)
            if utt.find("вопрос о мисис") >= 0:
                utt = utt[15:]
            if utt.find("о мисис") >= 0:
                utt = utt[8:]
            if utt.find("мисис") >= 0:
                utt = utt[6:]
            if utt.find("есть вопрос") >= 0:
                utt = utt[12:]
            if utt.find("хочу задать вопрос") >= 0:
                utt = utt[19:]
            if utt.find("робот") >= 0:
                utt = utt[6:]
            if utt.find("answer") >= 0:
                utt = utt[7:]
            if utt.find("миссис") >= 0:
                utt = utt[7:]
            print('Setting up client to connect to a local mycroft instance')
            if self.send_eye_check():
                logging.info("Персона обнаружена")
                logging.info("вопрос для модели: " + utt)
                r = self.voa_text(utt)
            else:
                logging.info("Персона потеряна")
                r = self.to_person
        except (Exception, ConnectionError) as err:
            logging.error("Произошела ошибка " + str(err))
            r = self.error_message
        self.speak(r)

    def ask_gpt(self, question):
        if question:
            question = question + ' университет МИСИС'
            messages = [{"role": "user", "content": question}]
            chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=messages
            )
        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")

    # @intent_handler(AdaptIntent()
    #                 .one_of("привет", "здравствуй", "добрый день", "добрый вечер", "приветствую", "доброе утро", "здравствуйте"))
    @intent_file_handler('misis.about.hi')
    def say_hello(self):
        try:
            if self.send_eye_check():
                self.speak("рад видеть вас")
            else:
                self.speak(self.to_person)
        except (Exception, ConnectionError) as err:
            logging.error("Произошела ошибка " + str(err))
            self.speak(self.error_message)


    # @intent_handler(AdaptIntent()
    #                 .one_of("пока", "до свидания"))
    @intent_file_handler('misis.about.bye')
    def say_bye(self):
        try:
            if self.send_eye_check():
                self.speak("До свидания, было приятно пообщаться")
            else:
                self.speak(self.to_person)
        except (Exception, ConnectionError) as err:
            logging.error("Произошела ошибка " + str(err))
            self.speak(self.error_message)

    def send_eye_check(self):
        # logging.info("Отправляем сообщенрие-проверку")
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.connect((self.TCP_IP, self.TCP_PORT))
        # s.sendall(bytes(self.MESSAGE, encoding="utf-8"))
        # data = s.recv(self.BUFFER_SIZE).decode('utf-8')
        # logging.info("Ответ зрения:" + str(data))
        # s.close()
        # if(str(data) == '1'):
        #     return True
        # if(str(data) == '0'):
        #     return False
        # else:
        #     raise Exception('Неизвестный тип сообщения от зрения')
        # return data
        return 'yes'

    def _load_model(self, config_path: str) -> Chainer:
        """Load or train deeppavlov model."""
        model_config = read_json(configs.doc_retrieval.ru_ranker_tfidf_wiki)
        model_config["dataset_reader"]["data_path"] = join(abspath(dirname(__file__)), "data", "misis_site")
        model_config["dataset_reader"]["dataset_format"] = "txt"
        model_config["train"]["batch_size"] = 1

        try:
            doc_retrieval = train_model(model_config)
            squad = build_model(join(abspath(dirname(__file__)), "data", "squad_ru_bert.json"), download=True)
            self._predictor = build_model(join(abspath(dirname(__file__)), "data", "ru_odqa_infer_wiki.json"), download=False)
        except FileNotFoundError:
            logging.warning("Невозможно загрузить модель")
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
        answer = predictor([question])

        logging.info("Полученный ответ: " + answer)
        if not answer or answer == '':
            answer = default_answer

        return answer


def create_skill():
    return AboutMisis()

# if __name__ == '__main__':
#     a = AboutMisis()
#     utt = 'направления'
#     r = a.voa_text(utt)
#     print(r)
