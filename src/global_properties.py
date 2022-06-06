import json

properties = None


def get_properties():

    global properties

    if properties is None:
        print(f'=============================')
        print(f'== load properties of file ==')
        print(f'=============================')

        f = open('C:\\dev\\workspaceMateus\\gaitule\\properties\\properties.json')
        properties = json.load(f)
        f.close()

    return properties

