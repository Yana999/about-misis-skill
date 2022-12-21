from mycroft import MycroftSkill, intent_file_handler


class AboutMisis(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self)

    @intent_file_handler('misis.about.intent')
    def handle_misis_about(self, message):
        self.speak_dialog('misis.about')


def create_skill():
    return AboutMisis()

