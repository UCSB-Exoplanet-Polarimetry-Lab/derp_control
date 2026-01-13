from derpy.motion import ZaberConnection, BaseZaberStage

con = ZaberConnection("COM4")
psg = BaseZaberStage(con, device=2)
psa = BaseZaberStage(con, device=3)

psg.step(90)
psa.step(90)