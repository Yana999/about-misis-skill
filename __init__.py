from mycroft import MycroftSkill, intent_file_handler
import requests



class AboutMisis(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self)
        url = "http://misisrobot.live:8000/"

    @intent_file_handler('misis.about.intent')
    def handle_misis_about(self, message):
        utt = message.data.get('utterance')
        # url = "http://misisrobot.live:8000/text-ask-text"
        # payload = {'question': utt}
        # r = requests.get(url)
        # self.speak(r.text)
        utt = str(utt)
        utt = utt.lower()
        if(utt.find("вопрос о МИСиС") >= 0):
            utt = utt[14:]
        if (utt.find("о МИСиС") >= 0):
            utt = utt[7:]
        if (utt.find("МИСиС") >= 0):
            utt = utt[5:]
        if (utt.find("есть вопрос") >= 0):
            utt = utt[11:]
        if (utt.find("хочу задать вопрос") >= 0):
            utt = utt[18:]
        if (utt.find("робот") >= 0):
            utt = utt[5:]
        self.speek(utt)

def create_skill():
    return AboutMisis()

