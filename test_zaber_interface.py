from derpy.motion import BaseZaberStage, ZaberConnection

zaber = ZaberConnection("COM4")

# init stage
zaber1 = BaseZaberStage(zaber, 0)
zaber2 = BaseZaberStage(zaber, 1)

zaber1.step(25)
zaber2.step(25)
zaber1.close()
zaber2.close()

