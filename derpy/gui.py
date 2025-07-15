"""
A vibe-coded gui module to try and make image cropping a little easier
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk

class ImageSquareSelector:
    def __init__(self, root, image_array):
        self.root = root
        self.root.title("Image Square Selector - Select Two Regions")
        self.root.geometry("800x600")

        # Variables
        self.image_array = image_array
        self.image = None
        self.photo = None
        self.canvas = None
        self.square_size = 50
        self.square1_id = None
        self.square2_id = None
        self.active_square = None
        self.drag_data = {"x": 0, "y": 0}
        self.selected_areas = []

        self.setup_ui()
        self.load_image_from_array()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Square size input
        ttk.Label(control_frame, text="Square Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.size_var = tk.StringVar(value=str(self.square_size))
        size_entry = ttk.Entry(control_frame, textvariable=self.size_var, width=10)
        size_entry.pack(side=tk.LEFT, padx=(0, 10))
        size_entry.bind('<Return>', self.update_square_size)

        # Update size button
        update_btn = ttk.Button(control_frame, text="Update Size", command=self.update_square_size)
        update_btn.pack(side=tk.LEFT, padx=(0, 10))

        # OK button
        ok_btn = ttk.Button(control_frame, text="OK - Save Both Areas", command=self.save_selected_areas)
        ok_btn.pack(side=tk.RIGHT)

        # Status label
        self.status_var = tk.StringVar(value="Drag the red and blue squares to select two regions")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=(0, 10))

        # Canvas frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)

    def load_image_from_array(self):
        """Load image from numpy array and display it"""
        try:
            # Ensure the array is in the correct format
            if self.image_array.dtype != np.uint8:
                # Normalize if needed
                if self.image_array.max() <= 1.0:
                    self.image_array = (self.image_array * 255).astype(np.uint8)
                else:
                    self.image_array = self.image_array.astype(np.uint8)

            # Handle different array shapes
            if len(self.image_array.shape) == 2:
                # Grayscale image
                self.image = Image.fromarray(self.image_array, mode='L')
            elif len(self.image_array.shape) == 3:
                if self.image_array.shape[2] == 3:
                    # RGB image
                    self.image = Image.fromarray(self.image_array, mode='RGB')
                elif self.image_array.shape[2] == 4:
                    # RGBA image
                    self.image = Image.fromarray(self.image_array, mode='RGBA')
                else:
                    raise ValueError("Unsupported number of channels")
            else:
                raise ValueError("Unsupported array shape")

            # Create PhotoImage for display
            self.photo = ImageTk.PhotoImage(self.image)

            # Clear canvas and display image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Update canvas scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            # Create initial squares
            self.create_squares()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image from array: {str(e)}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )

        if file_path:
            try:
                # Load image with PIL
                self.image = Image.open(file_path)
                self.photo = ImageTk.PhotoImage(self.image)

                # Clear canvas and display image
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

                # Update canvas scroll region
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))

                # Create initial squares
                self.create_squares()
                self.status_var.set("Drag the red and blue squares to select two regions")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )

        if file_path:
            try:
                # Load image with PIL
                self.image = Image.open(file_path)
                self.photo = ImageTk.PhotoImage(self.image)

                # Clear canvas and display image
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

                # Update canvas scroll region
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))

                # Create initial squares
                self.create_squares()
                self.status_var.set("Drag the red and blue squares to select two regions")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def update_square_size(self, event=None):
        try:
            new_size = int(self.size_var.get())
            if new_size > 0:
                self.square_size = new_size
                self.create_squares()
            else:
                messagebox.showerror("Error", "Square size must be a positive integer")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for square size")

    def create_squares(self):
        # Delete existing squares
        if self.square1_id:
            self.canvas.delete(self.square1_id)
        if self.square2_id:
            self.canvas.delete(self.square2_id)

        # Create first square (red) at center-left
        x1 = self.canvas.canvasx(150)
        y1 = self.canvas.canvasy(200)

        self.square1_id = self.canvas.create_rectangle(
            x1, y1, x1 + self.square_size, y1 + self.square_size,
            outline='red', width=3, fill='', tags="square1"
        )

        # Create second square (blue) at center-right
        x2 = self.canvas.canvasx(350)
        y2 = self.canvas.canvasy(200)

        self.square2_id = self.canvas.create_rectangle(
            x2, y2, x2 + self.square_size, y2 + self.square_size,
            outline='blue', width=3, fill='', tags="square2"
        )

    def start_drag(self, event):
        # Check which square is being clicked
        item = self.canvas.find_closest(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))[0]

        if item == self.square1_id:
            self.active_square = self.square1_id
            self.drag_data["x"] = self.canvas.canvasx(event.x)
            self.drag_data["y"] = self.canvas.canvasy(event.y)
        elif item == self.square2_id:
            self.active_square = self.square2_id
            self.drag_data["x"] = self.canvas.canvasx(event.x)
            self.drag_data["y"] = self.canvas.canvasy(event.y)
        else:
            self.active_square = None

    def drag(self, event):
        if self.active_square:
            # Calculate delta
            dx = self.canvas.canvasx(event.x) - self.drag_data["x"]
            dy = self.canvas.canvasy(event.y) - self.drag_data["y"]

            # Move the active square
            self.canvas.move(self.active_square, dx, dy)

            # Update drag data
            self.drag_data["x"] = self.canvas.canvasx(event.x)
            self.drag_data["y"] = self.canvas.canvasy(event.y)

    def stop_drag(self, event):
        self.active_square = None

    def save_selected_areas(self):
        if self.image_array is None:
            messagebox.showerror("Error", "No image array provided")
            return

        if self.square1_id is None or self.square2_id is None:
            messagebox.showerror("Error", "Both squares must be present")
            return

        try:
            # Get coordinates for both squares
            coords1 = self.canvas.coords(self.square1_id)
            coords2 = self.canvas.coords(self.square2_id)

            img_height, img_width = self.image_array.shape[:2]
            selected_areas = []

            # Process both squares
            for i, coords in enumerate([coords1, coords2], 1):
                x1, y1, x2, y2 = coords

                # Ensure coordinates are within image bounds
                x1 = max(0, min(int(x1), img_width))
                y1 = max(0, min(int(y1), img_height))
                x2 = max(0, min(int(x2), img_width))
                y2 = max(0, min(int(y2), img_height))

                if x1 >= x2 or y1 >= y2:
                    messagebox.showerror("Error", f"Invalid selection area for square {i}")
                    return

                # Extract the area directly from the numpy array
                area_array = self.image_array[y1:y2, x1:x2]
                selected_areas.append(area_array)

                print(f"Square {i} - Shape: {area_array.shape}, Coordinates: ({x1}, {y1}) to ({x2}, {y2})")

            self.selected_areas = selected_areas

            messagebox.showinfo("Success",
                f"Both areas saved as numpy arrays!\n"
                f"Area 1 shape: {selected_areas[0].shape}\n"
                f"Area 2 shape: {selected_areas[1].shape}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save selected areas: {str(e)}")

    def get_selected_areas(self):
        """Returns both selected areas as a list of numpy arrays"""
        return self.selected_areas

def launch_image_selector(image_array):
    """Main function to launch the GUI with a numpy array"""
    root = tk.Tk()
    app = ImageSquareSelector(root, image_array)
    root.mainloop()
    return app.get_selected_areas()

# Example usage
if __name__ == "__main__":
    # Example: Create a sample numpy array (replace with your actual image array)
    # You can load an image and convert it to numpy array like this:
    # from PIL import Image
    # image = Image.open("your_image.jpg")
    # image_array = np.array(image)
    import matplotlib.pyplot as plt
    import ipdb
    from astropy.io import fits
    from pathlib import Path

    pth = Path.home() / "Downloads/derp-selected/air_wollaston1deg_intsrphere/calibration_data_2025-07-14_17-20-06.fits"

    hdul = fits.open(pth)
    sample_image = hdul["PSA_IMAGES"].data[0]

    # For demonstration, create a random RGB image
    # sample_image = np.random.randint(0, 256, (400, 600), dtype=np.uint8)

    # Launch the GUI
    selected_areas = launch_image_selector(sample_image)

    plt.figure()
    for i, area in enumerate(selected_areas):
        plt.subplot(1, 2, i+1)
        plt.imshow(area)
    plt.show()

    # The selected areas will be available after the GUI is closed
    if selected_areas and len(selected_areas) == 2:
        print(f"Area 1 shape: {selected_areas[0].shape}")
        print(f"Area 2 shape: {selected_areas[1].shape}")
        print(f"Data types: {selected_areas[0].dtype}, {selected_areas[1].dtype}")
        # You can now use both selected_areas numpy arrays for further processing
