import shutil
import os


def recreatePath (path, dry = False):
    print ("Recreating path ", path)
    if dry == True:
        return None

    try:
        shutil.rmtree (path)
    except:
        pass
    os.makedirs (path)
    pass
