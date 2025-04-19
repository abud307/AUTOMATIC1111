from modules import script_callbacks, shared

def on_app_started(demo, app):
    print("Pollinations.AI API extension loaded!")

script_callbacks.on_app_started(on_app_started)
