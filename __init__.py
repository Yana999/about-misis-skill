import logging

from mycroft import MycroftSkill, intent_file_handler
from tools import VOATools
from config import VOAConfig

class AboutMisis(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self)
        config = VOAConfig()  # type: ignore
        voaTools = VOATools(config)

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
        r = self.voaTools.voa_text(utt)
        logging.info("Полученный ответ: " + r.a)
        self.speak(r.json()['answer'])

def create_skill():
    return AboutMisis()

# if __name__ == '__main__':
#     utt = 'направления'
#     config = VOAConfig()  # type: ignore
#     voaTools = VOATools(config)
#     r = voaTools.voa_text(utt)
#     print(r)
