import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import random
import math
import heapq
from collections import defaultdict, Counter

import matplotlib.pyplot as plt


class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


class ImageProcessingApp:
    def __init__(self, image_path=None, mode="RGB"):
        self.image_path = image_path
        self.mode_var = mode
        self.original_image = None
        self.edited_image = None
    def __init__(self, root):
        self.root = root
        self.root.title("ImageWizard")

        dark_gray = "#363636"
        darker_gray = "#505050"

        self.style = ttk.Style()
        self.style.configure("TButton", background=dark_gray, foreground=darker_gray, padding=10)
        self.style.configure("TLabel", background=dark_gray, foreground="white", padding=10)
        self.style.configure("TFrame", background=dark_gray)
        self.style.configure("TRadiobutton", background=dark_gray, foreground="white")

        self.paned_window = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=1)

        self.control_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.control_frame, weight=1)

        self.image_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.image_frame, weight=3)

        self.image_display_frame = ttk.Frame(self.image_frame)
        self.image_display_frame.pack(fill=tk.BOTH, expand=1)

        self.load_button = ttk.Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.mode_var = tk.StringVar(value="RGB")
        self.rgb_radio = ttk.Radiobutton(self.control_frame, text="RGB", variable=self.mode_var, value="RGB")
        self.rgb_radio.grid(row=0, column=1, padx=5, pady=5)
        self.grayscale_radio = ttk.Radiobutton(self.control_frame, text="Grayscale", variable=self.mode_var,
                                               value="Grayscale")
        self.grayscale_radio.grid(row=0, column=2, padx=5, pady=5)

        self.create_filter_frame()
        self.create_noise_frame()
        self.create_edge_detection_frame()
        self.create_histogram_frame()
        self.create_interpolation_frame()
        self.create_coding_frame()
        self.create_enhancement_frame()
        self.create_dft_frame()

        self.image_label = ttk.Label(self.image_display_frame)
        self.image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.save_button = ttk.Button(self.image_frame, text="Save Image", command=self.save_image)
        self.save_button.pack(padx=10, pady=10, fill=tk.X)

        self.image_path = None
        self.original_image = None
        self.edited_image = None

    def create_filter_frame(self):
        self.filter_frame = ttk.LabelFrame(self.control_frame, text="Filters")
        self.filter_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.blur_button = ttk.Button(self.filter_frame, text="Blur Image", command=self.apply_blur)
        self.blur_button.grid(row=0, column=0, padx=5, pady=5)
        self.gaussian_filter_button = ttk.Button(self.filter_frame, text="Gaussian Filter",
                                                 command=self.apply_gaussian_filter)
        self.gaussian_filter_button.grid(row=0, column=1, padx=5, pady=5)
        self.averaging_blur_button = ttk.Button(self.filter_frame, text="Averaging Blur",
                                                command=self.apply_averaging_blur)
        self.averaging_blur_button.grid(row=0, column=2, padx=5, pady=5)
        self.median_filter_button = ttk.Button(self.filter_frame, text="Median Filter",
                                               command=self.apply_median_filter)
        self.median_filter_button.grid(row=0, column=3, padx=5, pady=5)
        self.min_filter_button = ttk.Button(self.filter_frame, text="Min Filter", command=self.apply_min_filter)
        self.min_filter_button.grid(row=0, column=4, padx=5, pady=5)
        self.max_filter_button = ttk.Button(self.filter_frame, text="Max Filter", command=self.apply_max_filter)
        self.max_filter_button.grid(row=0, column=5, padx=5, pady=5)
        self.adaptive_median_filter_button = ttk.Button(self.filter_frame, text="Adaptive Median Filter",
                                                        command=self.apply_adaptive_median_filter)
        self.adaptive_median_filter_button.grid(row=0, column=6, padx=5, pady=5)

    def create_noise_frame(self):
        self.noise_frame = ttk.LabelFrame(self.control_frame, text="Noise")
        self.noise_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.gaussian_noise_button = ttk.Button(self.noise_frame, text="Add Gaussian Noise",
                                                command=self.apply_gaussian_noise)
        self.gaussian_noise_button.grid(row=0, column=0, padx=5, pady=5)
        self.salt_pepper_noise_button = ttk.Button(self.noise_frame, text="Salt and Pepper Noise",
                                                   command=self.apply_salt_pepper_noise)
        self.salt_pepper_noise_button.grid(row=0, column=1, padx=5, pady=5)

    def create_edge_detection_frame(self):
        self.edge_detection_frame = ttk.LabelFrame(self.control_frame, text="Edge Detection")
        self.edge_detection_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.sobel_button = ttk.Button(self.edge_detection_frame, text="Sobel Operator",
                                       command=self.apply_sobel_operator)
        self.sobel_button.grid(row=0, column=0, padx=5, pady=5)
        self.laplacian_button = ttk.Button(self.edge_detection_frame, text="Laplacian Operator",
                                           command=self.apply_laplacian)
        self.laplacian_button.grid(row=0, column=1, padx=5, pady=5)
        self.roberts_button = ttk.Button(self.edge_detection_frame, text="Roberts Cross-Gradient Operators",
                                         command=self.apply_roberts_cross)
        self.roberts_button.grid(row=0, column=2, padx=5, pady=5)

    def create_histogram_frame(self):
        self.histogram_frame = ttk.LabelFrame(self.control_frame, text="Histogram")
        self.histogram_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.histogram_eq_button = ttk.Button(self.histogram_frame, text="Histogram Equalization",
                                              command=self.apply_histogram_equalization)
        self.histogram_eq_button.grid(row=0, column=0, padx=5, pady=5)
        self.histogram_spec_button = ttk.Button(self.histogram_frame, text="Histogram Specification",
                                                command=self.apply_histogram_specification)
        self.histogram_spec_button.grid(row=0, column=1, padx=5, pady=5)

    def create_interpolation_frame(self):
        self.interpolation_frame = ttk.LabelFrame(self.control_frame, text="Interpolation")
        self.interpolation_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.width_label = ttk.Label(self.interpolation_frame, text="Width:")
        self.width_label.grid(row=0, column=0, padx=5, pady=5)
        self.width_entry = ttk.Entry(self.interpolation_frame)
        self.width_entry.grid(row=0, column=1, padx=5, pady=5)
        self.height_label = ttk.Label(self.interpolation_frame, text="Height:")
        self.height_label.grid(row=0, column=2, padx=5, pady=5)
        self.height_entry = ttk.Entry(self.interpolation_frame)
        self.height_entry.grid(row=0, column=3, padx=5, pady=5)
        self.nearest_neighbor_button = ttk.Button(self.interpolation_frame, text="Nearest Neighbor Interpolation",
                                                  command=self.apply_nearest_neighbor_interpolation)
        self.nearest_neighbor_button.grid(row=0, column=4, padx=5, pady=5)
        self.bilinear_button = ttk.Button(self.interpolation_frame, text="Bilinear Interpolation",
                                          command=self.apply_bilinear_interpolation)
        self.bilinear_button.grid(row=1, column=4, padx=5, pady=5)

    def create_coding_frame(self):
        self.coding_frame = ttk.LabelFrame(self.control_frame, text="Coding")
        self.coding_frame.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.huffman_button = ttk.Button(self.coding_frame, text="Huffman Encoding",
                                         command=self.apply_huffman_encoding)
        self.huffman_button.grid(row=0, column=0, padx=5, pady=5)
        self.golomb_button = ttk.Button(self.coding_frame, text="Golomb Coding", command=self.apply_golomb_coding)
        self.golomb_button.grid(row=0, column=1, padx=5, pady=5)

    def create_enhancement_frame(self):
        self.enhancement_frame = ttk.LabelFrame(self.control_frame, text="Enhancement")
        self.enhancement_frame.grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.unsharp_masking_button = ttk.Button(self.enhancement_frame, text="Unsharp Masking",
                                                 command=self.apply_unsharp_masking)
        self.unsharp_masking_button.grid(row=0, column=0, padx=5, pady=5)
        self.highboost_filtering_button = ttk.Button(self.enhancement_frame, text="Highboost Filtering",
                                                     command=self.apply_highboost_filtering)
        self.highboost_filtering_button.grid(row=0, column=1, padx=5, pady=5)

    def create_dft_frame(self):
        self.dft_frame = ttk.LabelFrame(self.control_frame, text="Frequency Domain")
        self.dft_frame.grid(row=8, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.highpass_filter_button = ttk.Button(self.dft_frame, text="High-Pass Filter (DFT)",
                                                 command=self.apply_highpass_filter)
        self.highpass_filter_button.grid(row=0, column=0, padx=5, pady=5)

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = Image.open(self.image_path)
            self.display_image(self.original_image)

    def display_image(self, image):
        if isinstance(image, Image.Image):
            img = ImageTk.PhotoImage(image)
            self.image_label.configure(image=img)
            self.image_label.image = img
        else:
            messagebox.showerror("Error", "Invalid image format.")

    def save_image(self):
        if self.edited_image:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                                ("All files", "*.*")])
            if save_path:
                self.edited_image.save(save_path)
                messagebox.showinfo("Success", "Image saved successfully.")
        else:
            messagebox.showerror("Error", "No edited image to save.")

    def apply_blur(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                self.edited_image = self.blur_image(self.grayscale_image(self.original_image))
            else:
                self.edited_image = self.blur_image(self.original_image)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_gaussian_filter(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                self.edited_image = self.gaussian_filter(self.grayscale_image(self.original_image), sigma=4.0)
            else:
                self.edited_image = self.gaussian_filter(self.original_image, sigma=4.0)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_grayscale(self):
        if self.image_path:
            self.edited_image = self.grayscale_image(self.original_image)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_gaussian_noise(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                self.edited_image = self.add_gaussian_noise(self.grayscale_image(self.original_image), mean=0,
                                                            variance=0.01)
            else:
                self.edited_image = self.add_gaussian_noise(self.original_image, mean=2, variance=0.11)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_bilinear_interpolation(self):
        if self.image_path:
            try:
                new_width = int(self.width_entry.get())
                new_height = int(self.height_entry.get())
                image_array = np.array(self.original_image)
                resized_image_array = self.bilinear_interpolation(image_array, new_width, new_height)
                self.edited_image = Image.fromarray(resized_image_array)
                self.display_image(self.edited_image)
            except ValueError:
                messagebox.showerror("Error", "Invalid width or height.")
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_adaptive_median_filter(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                filtered_image_array = self.adaptive_median_filter(np.array(self.grayscale_image(self.original_image)))
            else:
                filtered_image_array = self.adaptive_median_filter(np.array(self.original_image.convert('L')))
            self.edited_image = Image.fromarray(filtered_image_array)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_laplacian(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                self.edited_image = self.laplacian_operator(self.grayscale_image(self.original_image))
            else:
                self.edited_image = self.laplacian_operator(self.original_image)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_averaging_blur(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                self.edited_image = self.average_blur(self.grayscale_image(self.original_image), kernel_size=10)
            else:
                self.edited_image = self.average_blur(self.original_image.convert('L'), kernel_size=10)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_salt_pepper_noise(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                self.edited_image = self.salt_and_pepper_noise(self.grayscale_image(self.original_image), prob=0.05)
            else:
                self.edited_image = self.salt_and_pepper_noise(self.original_image, prob=0.05)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def is_grayscale(self, image):
        image_array = np.array(image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            for row in image_array:
                for pixel in row:
                    r, g, b = pixel
                    if r != g or r != b:
                        return False
            return True
        else:
            return len(image_array.shape) == 2

    def apply_median_filter(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                self.edited_image = self.median_filter(self.grayscale_image(self.original_image), window_size=10)
            else:
                self.edited_image = self.median_filter(self.original_image.convert('L'), window_size=10)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_min_filter(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                self.edited_image = self.min_filter(self.grayscale_image(self.original_image), window_size=3)
            else:
                self.edited_image = self.min_filter(self.original_image.convert('L'), window_size=3)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_max_filter(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                self.edited_image = self.max_filter(self.grayscale_image(self.original_image), window_size=5)
            else:
                self.edited_image = self.max_filter(self.original_image.convert('L'), window_size=5)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_sobel_operator(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                image_array = np.array(self.grayscale_image(self.original_image))
            else:
                image_array = np.array(self.original_image.convert('L'))

            magnitude, direction = self.sobel_operator(image_array)
            self.edited_image = Image.fromarray(magnitude.astype('uint8'))
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_histogram_equalization(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                image_array = np.array(self.grayscale_image(self.original_image))
            else:
                image_array = np.array(self.original_image.convert('L'))

            equalized_image_array = self.histogram_equalization(image_array)
            self.edited_image = Image.fromarray(equalized_image_array)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_histogram_specification(self):
        source_path = filedialog.askopenfilename(title="Select Source Image")
        reference_path = filedialog.askopenfilename(title="Select Reference Image")

        if source_path and reference_path:
            source_image = Image.open(source_path).convert("L")
            reference_image = Image.open(reference_path).convert("L")

            source_array = np.array(source_image)
            reference_array = np.array(reference_image)

            specified_image_array = self.histogram_specification(source_array, reference_array)
            self.edited_image = Image.fromarray(specified_image_array)
            self.display_image(self.edited_image)
        else:
            messagebox.showerror("Error", "Source or reference image not loaded.")

    def apply_nearest_neighbor_interpolation(self):
        if self.image_path:
            try:
                new_width = int(self.width_entry.get())
                new_height = int(self.height_entry.get())
                image_array = np.array(self.original_image)
                resized_image_array = self.nearest_neighbor_interpolation(image_array, new_width, new_height)
                self.edited_image = Image.fromarray(resized_image_array)
                self.display_image(self.edited_image)
            except ValueError:
                messagebox.showerror("Error", "Invalid width or height.")
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_huffman_encoding(self):
        if self.image_path:
            if self.is_grayscale(self.original_image):
                image_array = np.array(self.grayscale_image(self.original_image))
            else:
                image_array = np.array(self.grayscale_image(self.original_image))

            flat_image = []
            for row in image_array:
                for pixel in row:
                    flat_image.append(pixel)

            frequencies = {}
            for pixel in flat_image:
                if pixel in frequencies:
                    frequencies[pixel] += 1
                else:
                    frequencies[pixel] = 1

            huffman_tree = self.build_huffman_tree(frequencies)
            huffman_codebook = self.generate_codes(huffman_tree)

            encoded_image = self.huffman_encode(flat_image, huffman_codebook)
            decoded_image = self.huffman_decode(encoded_image, huffman_tree, image_array.shape)

            self.edited_image = Image.fromarray(decoded_image.astype('uint8'))
            self.display_image(self.edited_image)

            original_size = len(flat_image) * 8
            encoded_size = len(encoded_image)
            compression_ratio = original_size / encoded_size

            messagebox.showinfo("Huffman Encoding", f"Original size: {original_size} bits\n"
                                                    f"Encoded size: {encoded_size} bits\n"
                                                    f"Compression ratio: {compression_ratio:.2f}")

            # Save the encoded string
            self.save_encoded_string(encoded_image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_unsharp_masking(self):
        if self.image_path:
            if self.mode_var == "Grayscale":
                self.edited_image = self.unsharp_masking(np.array(self.grayscale_image(self.original_image)))
            else:
                self.edited_image = self.unsharp_masking(np.array(self.original_image))
            self.display_image(Image.fromarray(self.edited_image))
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_highboost_filtering(self):
        if self.image_path:
            if self.mode_var == "Grayscale":
                self.edited_image = self.highboost_filtering(np.array(self.grayscale_image(self.original_image)))
            else:
                self.edited_image = self.highboost_filtering(np.array(self.original_image))
            self.display_image(Image.fromarray(self.edited_image))
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_roberts_cross(self):
        if self.image_path:
            if self.mode_var == "Grayscale":
                self.edited_image = self.roberts_cross(np.array(self.grayscale_image(self.original_image)))
            else:
                self.edited_image = self.roberts_cross(np.array(self.grayscale_image(self.original_image)))
            self.display_image(Image.fromarray(self.edited_image))
        else:
            messagebox.showerror("Error", "No image loaded.")

    def apply_highpass_filter(self):
        if self.image_path:
            if self.mode_var == "Grayscale":
                image_array = np.array(self.grayscale_image(self.original_image))
            else:
                image_array = np.array(self.grayscale_image(self.original_image))

            F = self.dft2d(image_array)
            F_filtered = self.high_pass_filter(F)


            reconstructed_image = []
            for row in range(len(F_filtered)):
                real_row = []
                for col in range(len(F_filtered[0])):
                    real_row.append(F_filtered[row][col].real)
                reconstructed_image.append(real_row)

            self.edited_image = Image.fromarray(np.array(reconstructed_image).astype('uint8'))
            self.display_image(self.edited_image)

            self.display_frequency_domain(F, F_filtered)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def display_frequency_domain(self, F, F_filtered):
        magnitude_spectrum = np.log(np.abs(F) + 1)
        magnitude_spectrum_filtered = np.log(np.abs(F_filtered) + 1)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title('Original Frequency Domain')
        shifted_magnitude_spectrum = self.fft_shift(magnitude_spectrum)
        plt.imshow(shifted_magnitude_spectrum, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Filtered Frequency Domain')
        shifted_magnitude_spectrum_filtered = self.fft_shift(magnitude_spectrum_filtered)
        plt.imshow(shifted_magnitude_spectrum_filtered, cmap='gray')
        plt.axis('off')

        plt.show()

    def fft_shift(self, magnitude_spectrum):
        rows, cols = magnitude_spectrum.shape
        shifted_spectrum = np.zeros_like(magnitude_spectrum)

        shifted_spectrum[:rows // 2, :cols // 2] = magnitude_spectrum[rows // 2:, cols // 2:]
        shifted_spectrum[:rows // 2, cols // 2:] = magnitude_spectrum[rows // 2:, :cols // 2]
        shifted_spectrum[rows // 2:, :cols // 2] = magnitude_spectrum[:rows // 2, cols // 2:]
        shifted_spectrum[rows // 2:, cols // 2:] = magnitude_spectrum[:rows // 2, :cols // 2]

        return shifted_spectrum

    def apply_golomb_coding(self):
        if self.image_path:
            m = 10
            if self.is_grayscale(self.original_image):
                image_array = np.array(self.grayscale_image(self.original_image))
            else:
                image_array = np.array(self.grayscale_image(self.original_image))

            encoded_image = self.golomb_encode(image_array, m)
            decoded_image = self.golomb_decode(encoded_image, m, image_array.shape)

            self.edited_image = Image.fromarray(np.array(decoded_image, dtype=np.uint8))
            self.display_image(self.edited_image)

            original_size = image_array.size * 8  # 8 bits per pixel
            encoded_size = len(encoded_image)
            compression_ratio = original_size / encoded_size

            messagebox.showinfo("Golomb Coding", f"Original size: {original_size} bits\n"
                                                 f"Encoded size: {encoded_size} bits\n"
                                                 f"Compression ratio: {compression_ratio:.2f}")
        else:
            messagebox.showerror("Error", "No image loaded.")

    def golomb_encode(self, image_array, m):
        flat_image = []
        for row in image_array:
            for pixel in row:
                flat_image.append(pixel)
        encoded_image = []

        for value in flat_image:
            q = value // m
            r = value % m
            unary = '1' * q + '0'
            binary = self.to_binary(r, math.ceil(math.log2(m)))
            encoded_image.append(unary + binary)

        return ''.join(encoded_image)

    def to_binary(self, value, bits):
        binary = ''
        while value > 0:
            binary = str(value % 2) + binary
            value = value // 2
        while len(binary) < bits:
            binary = '0' + binary
        return binary

    def golomb_decode(self, encoded_image, m, shape):
        decoded_image = []
        i = 0
        while i < len(encoded_image):
            q = 0
            while encoded_image[i] == '1':
                q += 1
                i += 1
            i += 1  # skip the '0'

            r_bits = int(math.ceil(math.log2(m)))
            r = self.from_binary(encoded_image[i:i + r_bits])
            i += r_bits

            decoded_value = q * m + r
            decoded_image.append(decoded_value)

        return self.reshape(decoded_image, shape)

    def from_binary(self, binary_str):
        value = 0
        for char in binary_str:
            value = value * 2 + int(char)
        return value

    def reshape(self, flat_list, shape):
        reshaped = []
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):
                row.append(flat_list[i * shape[1] + j])
            reshaped.append(row)
        return np.array(reshaped)

    def blur_image(self, image):
        image_array = np.array(image)

        if len(image_array.shape) == 2:
            image_array = np.stack((image_array,) * 3, axis=-1)

        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
        blurred_image_array = np.copy(image_array)

        pad_size = kernel_size // 2
        padded_image = np.pad(image_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                for k in range(3):
                    blurred_image_array[i, j, k] = np.sum(
                        padded_image[i:i + kernel_size, j:j + kernel_size, k] * kernel)

        return Image.fromarray(blurred_image_array.astype('uint8'))

    def grayscale_image(self, image):
        image_array = np.array(image)
        grayscale_image_array = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
        grayscale_image_array = grayscale_image_array.astype('uint8')
        return Image.fromarray(grayscale_image_array)

    def generate_gaussian_noise(self, mean, variance):
        u = random.random()
        term1 = 1 / (math.sqrt(2 * math.pi * variance))
        term2 = math.exp((-1 / (2 * variance)) * ((u - mean) ** 2))
        return term1 * term2

    def add_gaussian_noise(self, image, mean=0, variance=0.01):
        image_array = np.array(image)
        noisy_image_array = np.zeros_like(image_array, dtype=np.uint8)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                for k in range(image_array.shape[2]):
                    u1 = 1 - np.random.rand()
                    u2 = 1 - np.random.rand()
                    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
                    noise = mean + z * math.sqrt(variance) * 255

                    noisy_pixel_value = image_array[i, j, k] + noise
                    noisy_image_array[i, j, k] = np.clip(noisy_pixel_value, 0, 255)

        return Image.fromarray(noisy_image_array)

    def laplacian_operator(self, image):
        image_array = self.convert_to_grayscale(image)

        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])

        laplacian_image_array = np.zeros_like(image_array, dtype=np.float64)

        pad_size = kernel.shape[0] // 2
        padded_image = self.pad_image(image_array, pad_size)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                region = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
                laplacian_value = np.sum(region * kernel)
                laplacian_image_array[i, j] = laplacian_value

        laplacian_image_array = np.clip(laplacian_image_array, 0, 255).astype('uint8')

        return Image.fromarray(laplacian_image_array)

    def convert_to_grayscale(self, image):
        grayscale_image_array = np.zeros((image.height, image.width), dtype=np.uint8)
        for i in range(image.height):
            for j in range(image.width):
                grayscale_image_array[i, j] = int(sum(image.getpixel((j, i))) / 3)
        return grayscale_image_array

    def pad_image(self, image_array, pad_size):
        padded_image = np.zeros((image_array.shape[0] + 2 * pad_size, image_array.shape[1] + 2 * pad_size),
                                dtype=np.uint8)
        padded_image[pad_size:image_array.shape[0] + pad_size, pad_size:image_array.shape[1] + pad_size] = image_array
        return padded_image

    def gaussian_filter(self, image, sigma):
        image_array = np.array(image.convert('L'))
        gaussian_blur = GaussianBlur(image_array)
        kernel = gaussian_blur.gaussian_kernel(sigma)
        blurred_image_array = gaussian_blur.apply_blur(kernel)
        return Image.fromarray(blurred_image_array)

    def calculate_mean(self, region):
        total = 0
        count = 0
        for row in region:
            for value in row:
                total += value
                count += 1
        return total // count

    def average_blur(self, image, kernel_size):
        image_array = np.array(image)

        pad_size = kernel_size // 2
        padded_image = self.pad_image(image_array, pad_size)
        output_image_array = np.zeros_like(image_array)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                output_image_array[i, j] = self.calculate_mean(region)

        return Image.fromarray(output_image_array.astype('uint8'))

    def salt_and_pepper_noise(self, image, prob):
        image_array = np.array(image)
        output = image_array.copy()
        if len(image_array.shape) == 2:
            black = 0
            white = 255
        else:
            colorspace = image_array.shape[2]
            if colorspace == 3:
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')

        probs = np.random.random(output.shape[:2])
        output[probs < (prob / 2)] = black
        output[probs > 1 - (prob / 2)] = white
        return Image.fromarray(output)

    def calculate_median(self, neighborhood):
        flat_neighborhood = neighborhood.flatten()
        sorted_neighborhood = np.sort(flat_neighborhood)
        middle = len(sorted_neighborhood) // 2
        if len(sorted_neighborhood) % 2 == 0:
            median_value = (sorted_neighborhood[middle - 1] + sorted_neighborhood[middle]) / 2.0
        else:
            median_value = sorted_neighborhood[middle]
        return median_value

    def median_filter(self, image, window_size):
        image_array = np.array(image.convert('L'))
        filtered_image_array = np.zeros_like(image_array)

        half_window = window_size // 2

        height, width = image_array.shape

        for y in range(height):
            for x in range(width):
                y1 = max(y - half_window, 0)
                y2 = min(y + half_window + 1, height)
                x1 = max(x - half_window, 0)
                x2 = min(x + half_window + 1, width)

                neighborhood = image_array[y1:y2, x1:x2]

                median_value = self.calculate_median(neighborhood)

                filtered_image_array[y, x] = median_value

        return Image.fromarray(filtered_image_array)

    def min_filter(self, image, window_size):
        image_array = np.array(image)
        padded_image = np.pad(image_array, ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)), mode='constant')
        min_filtered_image = np.zeros_like(image_array)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                min_filtered_image[i, j] = np.min(padded_image[i:i + window_size, j:j + window_size])

        return Image.fromarray(min_filtered_image.astype('uint8'))

    def max_filter(self, image, window_size):
        image_array = np.array(image)
        padded_image = np.pad(image_array, ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)), mode='constant')
        max_filtered_image = np.zeros_like(image_array)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                max_filtered_image[i, j] = np.max(padded_image[i:i + window_size, j:j + window_size])

        return Image.fromarray(max_filtered_image.astype('uint8'))

    def rgb_to_grayscale(self, pixel):
        return sum(pixel) // 3

    def adaptive_median_filter(self, image, max_kernel_size=7):
        def get_median(window):
            sorted_values = sorted(window.flatten())  # Flatten window and sort values
            length = len(sorted_values)
            if length % 2 == 0:  # If even number of elements
                return (sorted_values[length // 2 - 1] + sorted_values[length // 2]) / 2
            else:  # If odd number of elements
                return sorted_values[length // 2]

        def get_max(window):
            max_val = window[0, 0]
            for i in range(window.shape[0]):
                for j in range(window.shape[1]):
                    if window[i, j] > max_val:
                        max_val = window[i, j]
            return max_val

        def get_min(window):
            min_val = window[0, 0]
            for i in range(window.shape[0]):
                for j in range(window.shape[1]):
                    if window[i, j] < min_val:
                        min_val = window[i, j]
            return min_val

        def adaptive_filter(image, x, y, s_max):
            s = 1
            while s <= s_max:
                half_s = s // 2
                window = image[max(0, x - half_s):min(image.shape[0], x + half_s + 1),
                         max(0, y - half_s):min(image.shape[1], y + half_s + 1)]
                z_min = get_min(window)
                z_max = get_max(window)
                z_med = get_median(window)
                if z_min < z_med < z_max:
                    if z_min < image[x, y] < z_max:
                        return image[x, y]
                    else:
                        return z_med
                s += 2
            return z_med

        filtered_image = np.zeros_like(image)
        s_max = max_kernel_size

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                filtered_image[i, j] = adaptive_filter(image, i, j, s_max)

        return filtered_image

    def sobel_operator(self, image):
        Gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Gy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

        rows, cols = image.shape
        gradient_magnitude = np.zeros((rows, cols))
        gradient_direction = np.zeros((rows, cols))

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                region = image[i - 1:i + 2, j - 1:j + 2]
                gx = np.sum(Gx * region)
                gy = np.sum(Gy * region)
                gradient_magnitude[i, j] = np.sqrt(gx ** 2 + gy ** 2)
                gradient_direction[i, j] = np.arctan2(gy, gx)

        return gradient_magnitude, gradient_direction

    def histogram_equalization(self, image):
        histogram = np.zeros(256, dtype=int)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                value = image[i, j]
                histogram[value] += 1

        cdf = np.zeros_like(histogram, dtype=float)
        cdf[0] = histogram[0]
        for i in range(1, len(histogram)):
            cdf[i] = cdf[i - 1] + histogram[i]

        cdf_min = cdf.min()
        cdf_max = cdf.max()
        cdf_normalized = (cdf - cdf_min) * 255 / (cdf_max - cdf_min)
        cdf_normalized = cdf_normalized.astype('uint8')

        equalized_image = cdf_normalized[image]
        return equalized_image

    def custom_cumsum(self, array):
        cumsum = np.zeros_like(array, dtype=float)
        cumsum[0] = array[0]
        for i in range(1, len(array)):
            cumsum[i] = cumsum[i - 1] + array[i]
        return cumsum

    def histogram_specification(self, source_img, reference_img):
        source = np.array(source_img)
        reference = np.array(reference_img)

        source_hist = self.calculate_histogram(source)
        reference_hist = self.calculate_histogram(reference)

        source_cdf = self.calculate_cdf(source_hist)
        reference_cdf = self.calculate_cdf(reference_hist)

        source_cdf_normalized = self.normalize_cdf(source_cdf)
        reference_cdf_normalized = self.normalize_cdf(reference_cdf)

        lookup_table = np.zeros(256)

        for g_i in range(256):
            g_j = 0
            while g_j < 256 and source_cdf_normalized[g_i] > reference_cdf_normalized[g_j]:
                g_j += 1
            lookup_table[g_i] = g_j

        specified_img = np.zeros_like(source)
        for i in range(source.shape[0]):
            for j in range(source.shape[1]):
                specified_img[i, j] = lookup_table[source[i, j]]

        return specified_img

    def calculate_histogram(self, image):
        histogram = np.zeros(256, dtype=int)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                intensity = image[i, j]
                histogram[intensity] += 1

        return histogram

    def calculate_cdf(self, histogram):
        cdf = np.zeros_like(histogram, dtype=float)
        cdf[0] = histogram[0]
        for i in range(1, len(histogram)):
            cdf[i] = cdf[i - 1] + histogram[i]
        return cdf

    def normalize_cdf(self, cdf):
        cdf_min = cdf.min()
        cdf_max = cdf.max()
        return (cdf - cdf_min) * 255 / (cdf_max - cdf_min)
################################################################
    def nearest_neighbor_interpolation(self, image, new_width, new_height):
        orig_height, orig_width = image.shape[:2]
        row_ratio = orig_height / new_height
        col_ratio = orig_width / new_width
        new_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)

        for new_row in range(new_height):
            for new_col in range(new_width):
                orig_row = int(new_row * row_ratio)
                orig_col = int(new_col * col_ratio)
                new_image[new_row, new_col] = image[orig_row, orig_col]

        return new_image

    def bilinear_interpolation(self, image, new_width, new_height):
        orig_height, orig_width = image.shape[:2]
        new_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

        row_scale = orig_height / new_height
        col_scale = orig_width / new_width

        for new_row in range(new_height):
            for new_col in range(new_width):
                orig_row = new_row * row_scale
                orig_col = new_col * col_scale

                r = int(orig_row)
                c = int(orig_col)

                r1 = min(r + 1, orig_height - 1)
                c1 = min(c + 1, orig_width - 1)

                fr = orig_row - r
                fc = orig_col - c

                top_left = image[r, c]
                top_right = image[r, c1]
                bottom_left = image[r1, c]
                bottom_right = image[r1, c1]

                top = (1 - fc) * top_left + fc * top_right
                bottom = (1 - fc) * bottom_left + fc * bottom_right
                pixel = (1 - fr) * top + fr * bottom

                new_image[new_row, new_col] = np.clip(pixel, 0, 255)

        return new_image
#########################################################################
    def build_huffman_tree(self, frequencies):
        heap = MinHeap()

        for char, freq in frequencies.items():
            heap.push(HuffmanNode(char, freq))

        while len(heap.heap) > 1:
            node1 = heap.pop()
            node2 = heap.pop()
            merged = HuffmanNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heap.push(merged)

        return heap.heap[0]

    def generate_codes(self, node, prefix="", codebook={}):
        if node:
            if node.char is not None:
                codebook[node.char] = prefix
            self.generate_codes(node.left, prefix + "0", codebook)
            self.generate_codes(node.right, prefix + "1", codebook)
        return codebook

    def reshaping(self, array, new_shape):
        reshaped_array = [[0] * new_shape[1] for _ in range(new_shape[0])]
        index = 0
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                reshaped_array[i][j] = array[index]
                index += 1
        return reshaped_array

    def save_encoded_string(self, encoded_string):
        save_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if save_path:
            with open(save_path, 'w') as file:
                file.write(encoded_string)
            messagebox.showinfo("Success", "Encoded string saved successfully.")

    def huffman_encode(self, flat_image, codebook):
        encoded_image = ""
        for pixel in flat_image:
            encoded_image += codebook[pixel]
        return encoded_image

    def huffman_decode(self, encoded_image, huffman_tree, shape):
        current_node = huffman_tree
        decoded_image = []

        for bit in encoded_image:
            current_node = current_node.left if bit == '0' else current_node.right
            if current_node.char is not None:
                decoded_image.append(current_node.char)
                current_node = huffman_tree

        return np.array(decoded_image).reshape(shape)

    def gaussian_kernel(self, size, sigma=1):
        k = size // 2
        x, y = np.mgrid[-k:k + 1, -k:k + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
        return g

    def gaussian_kernel(self, size, sigma):
        kernel = np.zeros((size, size))
        center = size // 2
        constant = 1 / (2 * np.pi * sigma ** 2)
        for i in range(size):
            for j in range(size):
                distance_sq = (i - center) ** 2 + (j - center) ** 2
                kernel[i, j] = constant * np.exp(-distance_sq / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        return kernel

    def apply_filter(self, image, kernel):
        kernel_size = kernel.shape[0]
        padding = kernel_size // 2
        image_shape = image.shape
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
        filtered_image = np.zeros_like(image, dtype=np.float64)

        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                filtered_image[i, j] = np.sum(kernel * padded_image[i:i + kernel_size, j:j + kernel_size])

        return filtered_image.astype(np.uint8)

    def unsharp_masking(self, image, kernel_size=5, sigma=1, amount=1.0):
        kernel = self.gaussian_kernel(kernel_size, sigma)
        if image.ndim == 2:
            blurred = self.apply_filter(image, kernel)
            unsharp_mask = image - blurred
            enhanced_image = image + amount * unsharp_mask
        elif image.ndim == 3:
            enhanced_image = np.zeros_like(image)
            for i in range(3):
                blurred = self.apply_filter(image[:, :, i], kernel)
                unsharp_mask = image[:, :, i] - blurred
                enhanced_image[:, :, i] = image[:, :, i] + amount * unsharp_mask
        return np.clip(enhanced_image, 0, 255).astype(np.uint8)

    def highboost_filtering(self, image, kernel_size=5, sigma=1, k=1.5):
        kernel = self.gaussian_kernel(kernel_size, sigma)
        if image.ndim == 2:
            blurred = self.apply_filter(image, kernel)
            unsharp_mask = image - blurred
            highboost_image = image + k * unsharp_mask
        elif image.ndim == 3:
            highboost_image = np.zeros_like(image)
            for i in range(3):
                blurred = self.apply_filter(image[:, :, i], kernel)
                unsharp_mask = image[:, :, i] - blurred
                highboost_image[:, :, i] = image[:, :, i] + k * unsharp_mask
        return np.clip(highboost_image, 0, 255).astype(np.uint8)

    def roberts_cross(self, image):
        Gx = np.array([[1, 0], [0, -1]])
        Gy = np.array([[0, 1], [-1, 0]])

        image = image.astype(np.int32)
        rows, cols = image.shape
        gradient_magnitude = np.zeros((rows, cols))

        for i in range(rows - 1):
            for j in range(cols - 1):
                gx = np.sum(Gx * image[i:i + 2, j:j + 2])
                gy = np.sum(Gy * image[i:i + 2, j:j + 2])
                gradient_magnitude[i, j] = np.sqrt(gx ** 2 + gy ** 2)

        return np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    def dft2d(self, image):
        M, N = image.shape
        F = np.zeros((M, N), dtype=np.complex128)

        for u in range(M):
            for v in range(N):
                sum_val = 0
                for x in range(M):
                    for y in range(N):
                        sum_val += image[x, y] * np.exp(-2j * np.pi * ((u * x) / M + (v * y) / N))
                F[u, v] = sum_val

        return F

    def idft2d(self, F):
        M, N = F.shape
        image = np.zeros((M, N), dtype=np.float64)

        for x in range(M):
            for y in range(N):
                sum_val = 0
                for u in range(M):
                    for v in range(N):
                        sum_val += F[u, v] * np.exp(2j * np.pi * ((u * x) / M + (v * y) / N))
                image[x, y] = sum_val.real / (M * N)

        return image

    def high_pass_filter(self, F, cutoff=30):
        M, N = F.shape
        H = np.zeros((M, N))

        for u in range(M):
            for v in range(N):
                D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
                if D > cutoff:
                    H[u, v] = 1

        return F * H

    def build_huffman_tree(self, frequencies):
        heap = MinHeap()

        for char, freq in frequencies.items():
            heap.push(HuffmanNode(char, freq))

        while len(heap.heap) > 1:
            node1 = heap.pop()
            node2 = heap.pop()
            merged = HuffmanNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heap.push(merged)

        return heap.heap[0]


class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def push(self, node):
        self.heap.append(node)
        self.heapify_up(len(self.heap) - 1)

    def heapify_up(self, i):
        while i > 0 and self.heap[self.parent(i)].freq > self.heap[i].freq:
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def pop(self):
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify_down(0)
        return root

    def heapify_down(self, i):
        min_index = i
        left = self.left_child(i)
        right = self.right_child(i)

        if left < len(self.heap) and self.heap[left].freq < self.heap[min_index].freq:
            min_index = left
        if right < len(self.heap) and self.heap[right].freq < self.heap[min_index].freq:
            min_index = right

        if i != min_index:
            self.swap(i, min_index)
            self.heapify_down(min_index)


class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None


class GaussianBlur:
    def __init__(self, image):
        self.image = image

    def gaussian_kernel(self, sigma):
        size = int(math.ceil(3 * sigma))
        kernel = np.zeros((2 * size + 1, 2 * size + 1))

        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                kernel[i + size, j + size] = np.exp(-(i * i + j * j) / (2 * sigma * sigma))

        kernel /= np.sum(kernel)
        return kernel

    def apply_blur(self, kernel):
        kernel_size = kernel.shape[0]
        radius = kernel_size // 2
        image_padded = np.pad(self.image, ((radius, radius), (radius, radius)), mode='constant')
        blurred_image = np.zeros_like(self.image, dtype=np.float32)

        for i in range(blurred_image.shape[0]):
            for j in range(blurred_image.shape[1]):
                blurred_image[i, j] = np.sum(kernel * image_padded[i:i + kernel_size, j:j + kernel_size])

        return blurred_image.astype(np.uint8)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
