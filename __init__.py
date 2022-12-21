from mycroft import MycroftSkill, intent_file_handler


class AboutMisis(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self)

    @intent_file_handler('misis.about.intent')
    def handle_misis_about(self, message):
        utt = message.data.get('utterance')
        self.speak(utt)


def create_skill():
    return AboutMisis()

