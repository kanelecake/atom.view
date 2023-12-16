import os
from tkinter import Tk, Frame, Button, Canvas, Label, Listbox, END, ACTIVE, filedialog, Toplevel
from tkinter.ttk import Progressbar

from PIL import Image, ImageTk
from files import parse_dataset, parse_classes
from ultralytics import YOLO


class AtomView:
    def __init__(self):
        self.last_class_id = None
        self.updated_rectangles = None

        self.root = Tk()
        self.root.title("atomic.view")
        self.root.resizable(False, False)

        self.rectangles = {}
        self.paths = None
        self.dataset_info = None
        self.classes_list = None

        self.model = YOLO('./models/v13.pt')

        self.init_ui()

    def init_ui(self):
        self.create_panels()
        self.init_components()
        self.init_classlist()
        self.init_files_list()
        self.init_points_list()
        self.init_status_bar()
        self.root.mainloop()

    def get_prediction(self, path):
        results = self.model.predict(path, device='mps')
        result = {}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cords = box.xywh[0].cpu().numpy()  # get box coordinates in (left, top, right, bottom) format
                cls = int(box.cls[0].cpu().numpy())  # Convert the 0-dimensional array to an integer directly
                if cls in result:
                    result[cls].append([int(cords[0]), int(cords[1])])
                else:
                    result[cls] = [[int(cords[0]), int(cords[1])]]

        return result

    def create_panels(self):
        self.panel_left = Frame(self.root, width=180, padx=5, pady=10)
        self.panel_left.grid(row=0, column=0, padx=10, pady=0)

        self.panel_center = Frame(self.root, width=960, padx=5, pady=10)
        self.panel_center.grid(row=0, column=1, padx=10, pady=0)

    def init_status_bar(self):
        self.status_bar = Label(self.root, text="Готово!", bd=0, relief="sunken", anchor="w")
        self.status_bar.grid(row=1, column=0, columnspan=3, sticky="we", padx=15, pady=10)

        # Configure row and column weights to make the status bar expand horizontally
        self.root.rowconfigure(2, weight=1)  # Use row 2 instead of row 1
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)

    def update_classlist(self):
        self.classes_listbox.delete(0, END)
        self.classes_listbox.insert(END,
                                    *[f'{id}: {name}' for id, name in zip(self.classes_list[1], self.classes_list[0])])

    def update_files_list(self):
        self.paths_listbox.delete(0, END)
        self.paths_listbox.insert(END, *self.paths)

    def update_points_list(self, points):
        self.points_listbox.delete(0, END)
        self.points_listbox.insert(END, *points)

    def open_dataset_folder(self):
        folder_path = filedialog.askdirectory(title="Выберите папку с классами")
        if folder_path:
            self.load_data(folder_path)
            self.folder_path = folder_path
            self.update_classlist()
            self.update_files_list()
            active_points = self.dataset_info.get(self.paths[0], [])
            self.update_points_list(active_points)
            self.draw_rectangles_from_points(active_points)

            self.status_bar.config(text=f"Выбрана папка с датасетом: {folder_path}")
        else:
            # Display a warning if no folder is selected
            self.status_bar.config(text="Внимание! Папка не выбрана! ")

    def predict_photo(self):
        loading_window = Toplevel(self.root)
        loading_window.title("Loading...")
        loading_window.geometry("200x100")

        # Create a progress bar (indeterminate style for a spinner)
        progress_spinner = Progressbar(loading_window, mode='indeterminate')
        progress_spinner.pack(pady=20)
        progress_spinner.start()

        file_path = filedialog.askopenfilename(title="Select a File")
        self.set_image(file_path)
        prediction = self.get_prediction(file_path)
        keys = prediction.keys()

        for key in keys:
            self.draw_rectangle(prediction[key][0][0], prediction[key][0][1], int(key))

        progress_spinner.stop()
        loading_window.destroy()

    def init_components(self):
        self.open_classes = Button(self.panel_left, text="Открыть датасет", command=self.open_dataset_folder)
        self.open_classes.pack(pady=5)

        self.open_classes = Button(self.panel_left, text="Открыть изображение", command=self.predict_photo)
        self.open_classes.pack(pady=5)

        self.preview_canvas = Canvas(self.panel_center, width=960, height=600)

    def load_data(self, folder):
        metadata_folder = os.path.join(folder, 'metadata')
        self.dataset_info = parse_dataset(os.path.join(metadata_folder, 'set.cfg'))
        self.paths = list(self.dataset_info.keys())

    def init_classlist(self):
        self.classes_label = Label(self.panel_left, text="Список классов датасета:")
        self.classes_label.pack()

        self.classes_listbox = Listbox(self.panel_left)
        self.classes_listbox.pack(pady=5)

        self.classes_list = parse_classes(os.path.join('required/classes.cfg'))
        self.update_classlist()
        self.classes_listbox.bind('<<ListboxSelect>>', self.on_class_select)

    def on_class_select(self, evt):
        w = evt.widget
        if (index := w.curselection()):
            class_id = w.get(index[0]).split(': ')[0]

            for rectangle in (self.updated_rectangles or []):
                label, label_bg, border = rectangle
                color = 'black' if self.last_class_id == '0' else 'blue'
                self.preview_canvas.itemconfig(label_bg, fill=color, outline=color)
                self.preview_canvas.itemconfig(border, outline=color)

            if class_id in self.rectangles:
                for rectangle in self.rectangles[class_id]:
                    label, label_bg, border = rectangle
                    self.preview_canvas.itemconfig(label_bg, fill='green', outline='green')
                    self.preview_canvas.itemconfig(border, outline='green')

                self.updated_rectangles = self.rectangles.get(class_id, [])
                self.last_class_id = class_id

    def init_files_list(self):
        self.paths_label = Label(self.panel_left, text="Список файлов датасета:")
        self.paths_label.pack()

        self.paths_listbox = Listbox(self.panel_left)
        self.paths_listbox.pack(pady=5)
        self.paths_listbox.bind('<<ListboxSelect>>', self.on_select)

    def init_points_list(self):
        label = Label(self.panel_left, text="Список точек файла:")
        label.pack()

        self.points_listbox = Listbox(self.panel_left)
        self.points_listbox.pack(pady=5)

        self.points_listbox.delete(0, END)

    def draw_rectangles_from_points(self, points):
        for point in points:
            x, y, class_id = point
            self.draw_rectangle(int(x), int(y), class_id)

    def update_points_list(self, points):
        self.points_listbox.delete(0, END)
        self.points_listbox.insert(END, *points)
        self.draw_rectangles_from_points(points)

    def draw_rectangle(self, center_x, center_y, class_id):
        class_id = int(class_id)
        x1, y1, x2, y2 = center_x - 160 / 2, center_y - 160 / 2, center_x + 160 / 2, center_y + 160 / 2
        color = 'black' if class_id == 0 else 'blue'
        border = self.preview_canvas.create_rectangle(x1, y1, x2, y2, outline=color)
        text = self.classes_list[0][class_id]
        text_box_width, text_box_height = 60, 20
        label_bg = self.preview_canvas.create_rectangle(x1, y1, x1 + text_box_width, y1 + text_box_height, fill=color,
                                                        outline=color)
        label = self.preview_canvas.create_text(x1 + text_box_width / 2, y1 + text_box_height / 2, text=text,
                                                font=("Helvetica", 10), fill="white")
        if str(class_id) in self.rectangles:
            self.rectangles[str(class_id)].append([label, label_bg, border])
        else:
            self.rectangles[str(class_id)] = [[label, label_bg, border]]

    def set_image(self, path):
        self.image = Image.open(path)
        self.photo = ImageTk.PhotoImage(self.image)
        self.c_image = self.preview_canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.preview_canvas.grid(row=0, column=0)

    def on_select(self, evt):
        self.preview_canvas.delete('all')  # Clear previous rectangles
        self.rectangles = {}  # Clear stored rectangles

        w = evt.widget
        index = w.curselection()

        image_folder = os.path.join(self.folder_path, 'FRAMES')
        image_path = os.path.join(image_folder, w.get(index[0]))
        self.set_image(image_path.replace('\\', '/'))

        self.update_points_list(self.dataset_info.get(self.paths_listbox.get(ACTIVE), []))

    def update_rectangles(self, class_id):
        if self.updated_rectangles is not None:
            for rectangle in self.updated_rectangles:
                label, label_bg, border = rectangle
                color = 'black' if self.last_class_id == '0' else 'blue'
                self.preview_canvas.itemconfig(label_bg, fill=color, outline=color)
                self.preview_canvas.itemconfig(border, outline=color)
        if class_id in self.rectangles:
            for rectangle in self.rectangles[class_id]:
                label, label_bg, border = rectangle
                self.preview_canvas.itemconfig(label_bg, fill='green', outline='green')
                self.preview_canvas.itemconfig(border, outline='green')
                self.updated_rectangles = self.rectangles[class_id]
                self.last_class_id = class_id


app = AtomView()
