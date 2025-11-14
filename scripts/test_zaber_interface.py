from derpy.motion import BaseZaberStage, ZaberConnection

zaber = ZaberConnection("COM4")

# init stage
zaber1 = BaseZaberStage(zaber, 0) # PSA
zaber2 = BaseZaberStage(zaber, 1) # PSG

# home the thingy
pos = zaber1.get_current_position()
print(f"Current Position = {pos}deg")


zaber1.home()

pos = zaber1.get_current_position()
print(f"Current Position = {pos}deg")

#zaber1.step(185) #Set to 30


# zaber1.step(-180)

# pos = zaber1.get_current_position()
# print(f"Current Position = {pos}deg")

# for i in range(2):
#     zaber1.step(90)
#     pos = zaber1.get_current_position()
#     print(f"Current Position = {pos}deg")


zaber1.close()
#zaber2.close()

