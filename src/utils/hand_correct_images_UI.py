import os
import shutil
import re
from tkinter import Tk, Button, Label, filedialog, Canvas, simpledialog, messagebox
from PIL import Image, ImageTk


# ────────────────────────────────────────────────────────────────
#  Lista dodatkowych folderów i plików labels,
EXTRA_TARGETS = [
    # (folder_z_obrazami,  plik_labels)
    (r"C:\Users\krukw\PycharmProjects\Baumer_test\src\fake data\generated_fake_data_test_dataset",
     r"C:\Users\krukw\PycharmProjects\Baumer_test\src\fake data\generated_fake_data_test_dataset\labels.txt"),
    (r"C:\Users\krukw\PycharmProjects\Baumer_test\datasets\CV-X_dataset",
     r"C:\Users\krukw\PycharmProjects\Baumer_test\datasets\CV-X_dataset\labels.txt")
]

# ───────── AUTOSKIP, gdy etykiety są identyczne ─────────
AUTO_SKIP_IDENTICAL = True   # False → klasyczne ręczne przeglądanie
SECOND_LABELS_PATH = r"C:\Users\krukw\PycharmProjects\Baumer_test\src\generate_data\generated_fake_data_test_dataset_processed\code\read_labels.txt"
# ────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────


class ImageReviewerBase:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.index = 0
        self.root = Tk()
        self.root.title("Image Reviewer")

        self.label = Label(self.root, text="", font=("Arial", 14))
        self.label.pack()

        self.canvas = Canvas(self.root, width=800, height=600)
        self.canvas.pack()

        self.extra_label = Label(self.root, text="", font=("Arial", 10), fg="blue")
        self.extra_label.pack()

        self.extra2_label = Label(self.root, text="", font=("Arial", 10), fg="purple")
        self.extra2_label.pack()

        self.info_label = Label(self.root, text="", font=("Arial", 10), fg="gray")
        self.info_label.pack(pady=5)

        self.quit_button = Button(self.root, text="Zakończ", command=self.quit)
        self.quit_button.pack(pady=5)

        self.root.bind("<Key>", self.key_handler)

    def load_image(self):
        pass

    def display_image(self):
        image = self.current_image.copy()
        image.thumbnail((800, 600))
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(400, 300, image=self.tk_image)

    def quit(self):
        self.root.destroy()

    def key_handler(self, event):
        pass


class RotateReviewer(ImageReviewerBase):
    def __init__(self, folder_path):
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        self.image_files.sort()
        super().__init__(folder_path)
        self.info_label.config(text="Tryb obracania: [o] obróć, [p] pomiń, [←] poprzedni")
        self.load_image()
        self.root.mainloop()

    def load_image(self):
        if 0 <= self.index < len(self.image_files):
            image_path = os.path.join(self.folder_path, self.image_files[self.index])
            self.current_image = Image.open(image_path)
            self.display_image()
            self.label.config(text=self.image_files[self.index])
            self.extra_label.config(text="")
        else:
            self.label.config(text="Brak obrazów do wyświetlenia.")
            self.canvas.delete("all")

    def rotate_and_save(self):
        image_path = os.path.join(self.folder_path, self.image_files[self.index])
        rotated = self.current_image.rotate(180)
        rotated.save(image_path)
        self.next_image()

    def next_image(self):
        if self.index < len(self.image_files) - 1:
            self.index += 1
            self.load_image()
        else:
            self.label.config(text="Koniec obrazów.")
            self.canvas.delete("all")

    def previous_image(self):
        if self.index > 0:
            self.index -= 1
            self.load_image()

    def key_handler(self, event):
        if event.char.lower() == 'o':
            self.rotate_and_save()
        elif event.char.lower() == 'p':
            self.next_image()
        elif event.keysym == 'Left':
            self.previous_image()


class DeleteReviewer(ImageReviewerBase):
    """Tryb usuwania obrazów + synchronizacja z dodatkowymi repozytoriami."""
    def __init__(self, folder_path, labels_path, extra_targets=None):
        self.labels_path = labels_path
        self.extra_targets = extra_targets or []          # lista (folder, labels)

        with open(labels_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.image_files = [line.split()[0] for line in self.lines]

        # === przygotuj dodatkowe listy ===
        self.extra_lines = []
        for _, lbl in self.extra_targets:
            if os.path.exists(lbl):
                with open(lbl, 'r', encoding='utf-8') as f:
                    self.extra_lines.append([ln.strip() for ln in f if ln.strip()])
            else:
                self.extra_lines.append([])  # brak pliku = pusta lista

        # === wiersze z pliku do podglądu B ===
        if os.path.exists(SECOND_LABELS_PATH):
            with open(SECOND_LABELS_PATH, 'r', encoding='utf-8') as f:
                self.second_lines = [ln.strip() for ln in f if ln.strip()]
        else:
            self.second_lines = []

        super().__init__(folder_path)
        self.info_label.config(text="Tryb usuwania: [u] usuń, [p] pomiń, [←] poprzedni")
        self.load_image()
        self.root.mainloop()

    def load_image(self):
        # pętla, aby automatycznie omijać identyczne wiersze
        while True:
            if not (0 <= self.index < len(self.image_files)):
                self.label.config(text="Koniec obrazów.")
                self.extra_label.config(text="")
                self.extra2_label.config(text="")
                self.canvas.delete("all")
                return

            image_name = self.image_files[self.index]
            line_main  = self.lines[self.index]

            # szukamy odpowiadającego wiersza w second_lines
            second_line = next(
                (ln for ln in self.second_lines if ln.startswith(image_name)),
                ""
            )

            # == AUTOSKIP ==
            if AUTO_SKIP_IDENTICAL and second_line:
                # porównujemy część po nazwie pliku (etykietę właściwą)
                main_tail   = line_main.partition(' ')[2]
                second_tail = second_line.partition(' ')[2]
                if main_tail == second_tail:
                    self.index += 1       # pomijamy BEZ usuwania
                    continue              # sprawdź kolejny obraz
            # == end AUTOSKIP ==

            # wyświetlenie, jeśli nie pominięto
            image_path = os.path.join(self.folder_path, image_name)
            if os.path.exists(image_path):
                self.current_image = Image.open(image_path)
                self.display_image()
            else:
                self.canvas.delete("all")

            self.label.config(text=image_name)
            self.extra_label.config(text=line_main)
            self.extra2_label.config(text=second_line)
            break


    def delete_current(self):
        fname = self.image_files[self.index]
        try:
            # usuń plik główny
            main_img = os.path.join(self.folder_path, fname)
            if os.path.exists(main_img):
                os.remove(main_img)

            # usuń z głównego labels
            del self.lines[self.index]
            del self.image_files[self.index]
            self.save_labels(self.labels_path, self.lines)

            # usuń we WSZYSTKICH dodatkowych lokalizacjach
            for (folder, lbl_path), lines in zip(self.extra_targets, self.extra_lines):
                # plik
                img_p = os.path.join(folder, fname)
                if os.path.exists(img_p):
                    os.remove(img_p)
                # wiersz w labels
                lines[:] = [ln for ln in lines if not ln.startswith(fname)]
                self.save_labels(lbl_path, lines)

            # załaduj kolejny / poprzedni obraz
            if self.index >= len(self.image_files):
                self.index = len(self.image_files) - 1
            self.load_image()

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się usunąć: {e}")

    @staticmethod
    def save_labels(path, lines):
        with open(path, 'w', encoding='utf-8') as f:
            for ln in lines:
                f.write(ln + '\n')

    def next_image(self):
        if self.index < len(self.image_files) - 1:
            self.index += 1
            self.load_image()
        else:
            self.label.config(text="Koniec obrazów.")
            self.canvas.delete("all")

    def previous_image(self):
        if self.index > 0:
            self.index -= 1
            self.load_image()

    def key_handler(self, event):
        if event.char.lower() == 'u':
            self.delete_current()
        elif event.char.lower() == 'p':
            self.next_image()
        elif event.keysym == 'Left':
            self.previous_image()


def truncate_labels(file_path, num_chars_to_remove, output_path=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.rstrip('\n')  # usuń newline
        if len(line) >= num_chars_to_remove:
            new_line = line[:-num_chars_to_remove]
        else:
            new_line = ''
        new_lines.append(new_line + '\n')

    output = output_path if output_path else file_path
    with open(output, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"Zapisano poprawiony plik: {output}")


def truncate_labels_codes(file_path, date_prefix_len=11, output_path=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.rstrip('\n')

        # Rozdziel po spacji — pierwszy element to nazwa pliku
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            # jeśli coś jest nie tak z formatem linii, pomijamy lub zapisujemy pustą linię
            new_lines.append('\n')
            continue

        filename, rest = parts

        # Odcięcie pierwszych date_prefix_len znaków po nazwie pliku
        if len(rest) > date_prefix_len:
            remaining = rest[date_prefix_len:].strip()
        else:
            remaining = ''

        new_lines.append(f"{filename} {remaining}\n")

    output = output_path if output_path else file_path
    with open(output, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"Zapisano poprawiony plik: {output}")


def increment_image_numbers(file_path, offset, output_path=None):
    """Dodaje "offset" do numerów zdjęć w nazwach plików w pliku labels.txt.

    Przykład:
        img_1.jpg -> img_11.jpg przy offset=10
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.rstrip('\n')
        if not line.strip():
            new_lines.append('\n')
            continue

        parts = line.split(maxsplit=1)
        filename = parts[0]
        rest = parts[1] if len(parts) > 1 else ''

        match = re.match(r'(.*?)(\d+)(\.[^.]+)$', filename)
        if match:
            prefix, num_str, ext = match.groups()
            new_num = int(num_str) + offset
            new_filename = f"{prefix}{new_num}{ext}"
        else:
            # Jeśli nie udało się sparsować numeru, zostawiamy nazwę bez zmian
            new_filename = filename

        if rest:
            new_lines.append(f"{new_filename} {rest}\n")
        else:
            new_lines.append(f"{new_filename}\n")

    output = output_path if output_path else file_path
    with open(output, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"Zapisano zaktualizowany plik: {output}")


def main():
    mode = simpledialog.askstring(
        "Wybierz tryb",
        "Wpisz tryb: 'obracanie' lub 'usuwanie' lub 'tylkodaty' lub 'tylkokody' lub 'dodajnumery'"
    )

    if not mode:
        return

    mode = mode.lower()
    if mode not in ['obracanie', 'usuwanie', 'tylkodaty', 'tylkokody', 'dodajnumery']:
        messagebox.showerror("Błąd", "Nieprawidłowy tryb.")
        return

    if mode in ['obracanie', 'usuwanie']:
        folder = filedialog.askdirectory(title="Wybierz folder z obrazami")
        if not folder:
            return

    if mode == 'obracanie':
        RotateReviewer(folder)
    elif mode == 'usuwanie':
        labels_file = filedialog.askopenfilename(title="Wybierz plik labels.txt", filetypes=[("Text Files", "*.txt")])
        if not labels_file:
            return
        DeleteReviewer(folder, labels_file, extra_targets=EXTRA_TARGETS)
    elif mode == 'tylkodaty':
        labels_file = filedialog.askopenfilename(title="Wybierz plik labels.txt", filetypes=[("Text Files", "*.txt")])
        if not labels_file:
            return
        truncate_labels(labels_file, num_chars_to_remove=16, output_path=None)
    elif mode == 'tylkokody':
        labels_file = filedialog.askopenfilename(title="Wybierz plik labels.txt", filetypes=[("Text Files", "*.txt")])
        if not labels_file:
            return
        truncate_labels_codes(labels_file, date_prefix_len=11, output_path=None)
    elif mode == 'dodajnumery':
        labels_file = filedialog.askopenfilename(title="Wybierz plik labels.txt", filetypes=[("Text Files", "*.txt")])
        if not labels_file:
            return
        offset = simpledialog.askinteger("Podaj liczbę", "Podaj liczbę, którą chcesz dodać do numerów zdjęć:")
        if offset is None:
            return
        increment_image_numbers(labels_file, offset, output_path=None)


if __name__ == "__main__":
    main()
