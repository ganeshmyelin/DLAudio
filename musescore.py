from music21 import *
#us = environment.UserSettings()
#us.create()
us = environment.UserSettings()
print(us.getSettingsPath())
for key in sorted(us.keys()):
    print(key)

us['musicxmlPath'] = '/Applications/MuseScore 3.app/Contents/MacOS/mscore'
print(us['musicxmlPath'])