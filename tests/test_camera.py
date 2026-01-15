# tests the new Base Camera class written by Courtney Duong
import matplotlib.pyplot as plt
from derpy.base_camera import ZWOASI

# Init camera with defaults
if __name__ == "__main__":
    cam = ZWOASI(
        camera_index=0,
        fps=200,
        tint=10,
        conversion_gain=150,
        set_temperature=0,
        temp_tolerance=5,
        bit_depth=16
    )

    # Test methods in order
    print(f"Camera temperature = {cam.get_temperature()}")

    # print(f"Setting camera temperature = 25")
    # cam.set_temperature(25)
    # print(f"Camera temperature = {cam.get_temperature()}")

    frame = cam.take_many_images(num_frames=1)[0]

    # Frame is RGB
    print(frame.shape)
    plt.figure()
    plt.imshow(frame, vmin=0, vmax=2**16)
    plt.colorbar()
    plt.show()

