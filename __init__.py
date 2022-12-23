import logging

from mycroft import MycroftSkill, intent_file_handler
import requests

class AboutMisis(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self)

    @intent_file_handler('misis.about.intent')
    def handle_misis_about(self, message):
        utt = message.data.get('utterance')
        logging.info("тип utterance: " + str(type(utt)))
        utt = str(utt)
        logging.info("Полученный текст: " + utt)
        if(utt.find("вопрос о МИСиС") >= 0):
            utt = utt[15:]
        if (utt.find("о МИСиС") >= 0):
            utt = utt[8:]
        if (utt.find("МИСиС") >= 0):
            utt = utt[6:]
        if (utt.find("есть вопрос") >= 0):
            utt = utt[12:]
        if (utt.find("хочу задать вопрос") >= 0):
            utt = utt[19:]
        if (utt.find("робот") >= 0):
            utt = utt[6:]
        if (utt.find("answer") >= 0):
            utt = utt[7:]
        url = "http://misisrobot.live:8000/text-ask-text?question=" + utt
        r = requests.get(url)
        logging.info("Полученный ответ: " + r.text)
        self.speak(r.json()['answer'])

def create_skill():
    return AboutMisis()

# if __name__ == '__main__':
#     url = "http://misisrobot.live:8000/text-ask-text?question=" + "направления"
#     r = requests.get(url)
#     print(r.json()['answer'])
