import json
import logging
from os.path import join, abspath, dirname
from pathlib import Path
from typing import Optional
import socket
import time
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch
import difflib
from transformers import BertTokenizer, BertModel
from deeppavlov import Chainer, train_model, build_model
from deeppavlov.core.common.file import read_json
from mycroft import MycroftSkill, intent_handler
from nltk.corpus import stopwords
import re
import random
from pymorphy2 import MorphAnalyzer


class AboutMisis(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self, "AboutMISISSkill")
        self.ml_root_path = './data'
        self.tmp_dir = Path('./tmp_data')
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        # self._ml_path = join(abspath(dirname(__file__)), "data", "tfidf_logreg_autofaq_misis.json")
        self._ml_path = join(abspath(dirname(__file__)), 'lev_model', 'best_model_state.bin')
        self._predictor: Optional[Chainer] = None
        self.is_eye = False
        self.TCP_IP = '192.168.1.101'
        self.TCP_PORT = 5005
        self.BUFFER_SIZE = 128
        self.MESSAGE = json.dumps({'type': 'eye'})
        self.to_person = 'Пожалуйста, вернитесь в фокус зрения робота'
        self.error_message = 'Есть технический сбой в моих системах произвожу перезагрузку'
        self.pre_trained_model_ckpt = 'DeepPavlov/rubert-base-cased'
        self.device = torch.device('cpu')
        self.patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
        self.stopwords_ru = stopwords.words("russian")
        self.morph = MorphAnalyzer()
        self.tokenizer = None
        self.dataset = join(abspath(dirname(__file__)), 'data', '12042023_dataset_sort_change_tokenizer_fix_sort.xlsx')
        self.class_names = ['поступление - перевод', 'общежитие', 'учебная деятельность', 'внеучебная деятельность', 'документы', 'работа', 'финансы']

    @intent_handler('misis.about.intent.intent')
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
                r = self.make_ans(utt, self.dataset)
            else:
                logging.info("Персона потеряна")
                r = self.to_person
        except (Exception, ConnectionError) as err:
            logging.error("Произошела ошибка " + str(err))
            r = self.error_message
        self.speak(r)

    @intent_handler('misis.about.hi.intent')
    def say_hello(self):
        try:
            if self.send_eye_check():
                self.speak("рад видеть вас")
            else:
                self.speak(self.to_person)
        except (Exception, ConnectionError) as err:
            logging.error("Произошела ошибка " + str(err))
            self.speak(self.error_message)

    @intent_handler('misis.about.bye.intent')
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
        # if (str(data) == '1'):
        #     return True
        # if (str(data) == '0'):
        #     return False
        # else:
        #     raise Exception('Неизвестный тип сообщения от зрения')
        # return data
        return True

    def _load_model(self, config_path: str) -> Chainer:
        """Load or train deeppavlov model."""
        model_config = read_json(config_path)
        model_config["dataset_reader"]["data_url"] = join(abspath(dirname(__file__)), "data",
                                                          "07042023_dataset_sort_povtor.csv")
        model_config["metadata"]["variables"]["ROOT_PATH"] = (
            self.ml_root_path
        )
        try:
            self._predictor = build_model(model_config, load_trained=True)
        except FileNotFoundError:
            self._predictor = train_model(model_config, download=True)
        return self._predictor

    def load_model(self, config_path: Optional[str] = None):
        """Return deeppavlov model."""
        if config_path is None or config_path == self._ml_path:
            if self._predictor:
                return self._predictor

        if not config_path:
            config_path = self._ml_path
        predictor = self._load_lev_model(config_path)

        if config_path != self._ml_path:
            self._ml_path = config_path
        return predictor

    def load_tokenizer(self):
        if not self.tokenizer:
            self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_model_ckpt)
        return self.tokenizer

    def _load_lev_model(self, config_path: str):
        class_names = ['поступление - перевод', 'общежитие', 'учебная деятельность', 'внеучебная деятельность',
                       'документы', 'работа', 'финансы']
        myModel = SentimentClassifier(len(class_names))
        myModel.load_state_dict(torch.load(config_path, map_location=self.device))
        myModel = myModel.to(self.device)
        return myModel

    def lemmatize(self, doc):
        doc = re.sub(self.patterns, ' ', doc)
        tokens = []
        for token in doc.split():
            if token and token not in self.stopwords_ru:
                token = token.strip()
                token = self.morph.normal_forms(token)[0]

                tokens.append(token)
        return self.load_tokenizer().encode(" ".join(tokens), add_special_tokens=False)

    def make_ans(self,  question, dataset):
        encoded_review = self.load_tokenizer().encode_plus(question, max_length=512, add_special_tokens=True,
                                               return_token_type_ids=False, pad_to_max_length=True,
                                               return_attention_mask=True,
                                               truncation=True, return_tensors='pt')
        input_ids = encoded_review['input_ids'].to(self.device)
        attention_mask = encoded_review['attention_mask'].to(self.device)
        logging.info("Загрузка модели")
        predictor = self.load_model()
        output = predictor(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)

        logging.info("лематизация")
        tToken = self.lemmatize(question)
        tq = []
        index = 0
        probability = 0.00
        textQ = ''
        textA = ''
        questions = []
        # import main df
        df = pd.read_excel(dataset)
        logging.info("цикл")
        for i in range(0, len(df)):
            if (probability <= self.mySort(tToken, eval(str(df['tq_fix'][i]))) and df['label'][i] == int(prediction)):
                tq = eval(str(df['tq_fix'][i]))
                index = i
                textQ = df['q'][i]
                textA = df['a'][i]
                probability = self.mySort(tToken, eval(str(df['tq_fix'][i])))
                questions.append([tq, index, textQ, textA, probability])

        max_last_elem = max([que[-1] for que in questions])
        max_lists = [lst for lst in questions if lst[-1] == max_last_elem]
        rand = random.choice(max_lists)
        logging.info(f'{question} - Текст основного вопроса' + '\n' +
                         f'{self.class_names[prediction]} - ИИ класс' + '\n' +
                         f'{tToken} - Токенизированный текст основного вопроса' + '\n' +
                         f'{rand[4] * 100}% - Процент сходства с найденным вопросом' + '\n' +
                         f'{rand[1]} - Номер найденного вопроса' + '\n' +
                         f'{rand[2]} - Текст найденного вопроса' + '\n' +
                         f'{rand[0]} - Токенизированный найденный вопрос' + '\n' +
                         f'{rand[3]} - Ответ к найденному вопросу')

        return rand[3]

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
        logging.info("Полученный ответ: " + answer)
        logging.info("Score: ")
        logging.info(score)
        if not score:
            answer = default_answer
        return answer

    # make match
    def mySort(self, s1, s2):
        matcher = difflib.SequenceMatcher(None, s1, s2)
        return matcher.ratio()

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.pre_trained_model_ckpt = 'DeepPavlov/rubert-base-cased'
        self.bert = BertModel.from_pretrained(self.pre_trained_model_ckpt, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


def create_skill():
    return AboutMisis()

# if __name__ == '__main__':
#     a = AboutMisis()
#     utt = 'как подать ддокументы'
#     r = a.make_ans(utt, a.dataset)
#     print(r)
#     utt = 'как оплатить обучение'
#     r = a.make_ans(utt, a.dataset)
#     print(r)
