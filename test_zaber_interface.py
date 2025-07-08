from derpy.motion import BaseZaberStage

# init stage
zaber = BaseZaberStage("COM4", 0)

zaber.step(25)
zaber.close()

