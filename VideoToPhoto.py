import customtkinter as ctk
from tkinter import filedialog
import cv2
import os
import threading

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Настройка окна ---
        self.title("Раскадровка видео")
        self.geometry("700x450")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- Переменные состояния ---
        self.input_video_path = ctk.StringVar()
        self.output_folder_path = ctk.StringVar()

        # --- Виджеты ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # Поле для выбора видео
        self.label_input = ctk.CTkLabel(self, text="Видеофайл:")
        self.label_input.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        self.entry_input = ctk.CTkEntry(self, textvariable=self.input_video_path, state="readonly")
        self.entry_input.grid(row=0, column=1, padx=20, pady=(20, 10), sticky="ew")
        self.button_browse_input = ctk.CTkButton(self, text="Выбрать...", command=self.browse_video)
        self.button_browse_input.grid(row=0, column=2, padx=20, pady=(20, 10))

        # Поле для выбора папки
        self.label_output = ctk.CTkLabel(self, text="Папка для кадров:")
        self.label_output.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.entry_output = ctk.CTkEntry(self, textvariable=self.output_folder_path, state="readonly")
        self.entry_output.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        self.button_browse_output = ctk.CTkButton(self, text="Выбрать...", command=self.browse_folder)
        self.button_browse_output.grid(row=1, column=2, padx=20, pady=10)

        # Кнопка "Старт"
        self.start_button = ctk.CTkButton(self, text="Начать раскадровку", command=self.start_extraction_thread)
        self.start_button.grid(row=2, column=1, padx=20, pady=20)

        # Прогресс-бар
        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=3, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        
        # Окно для логов
        self.log_textbox = ctk.CTkTextbox(self, state="disabled")
        self.log_textbox.grid(row=4, column=0, columnspan=3, padx=20, pady=(10, 20), sticky="nsew")

    def log(self, message):
        """Добавляет сообщение в окно логов."""
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.configure(state="disabled")
        self.log_textbox.see("end") # Автопрокрутка

    def browse_video(self):
        """Открывает диалог для выбора видеофайла."""
        filepath = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
        )
        if filepath:
            self.input_video_path.set(filepath)
            self.log(f"Выбран видеофайл: {filepath}")

    def browse_folder(self):
        """Открывает диалог для выбора папки."""
        folderpath = filedialog.askdirectory(title="Выберите папку для сохранения кадров")
        if folderpath:
            self.output_folder_path.set(folderpath)
            self.log(f"Выбрана папка для вывода: {folderpath}")

    def start_extraction_thread(self):
        """Запускает извлечение кадров в отдельном потоке, чтобы GUI не зависал."""
        input_path = self.input_video_path.get()
        output_path = self.output_folder_path.get()

        if not input_path or not output_path:
            self.log("Ошибка: Укажите путь к видеофайлу и папку для сохранения.")
            return

        self.start_button.configure(state="disabled", text="В процессе...")
        self.progress_bar.set(0)
        
        # Запускаем извлечение в новом потоке
        thread = threading.Thread(target=self.extract_frames, args=(input_path, output_path))
        thread.daemon = True
        thread.start()

    def extract_frames(self, video_path, output_folder):
        """Основная логика извлечения кадров из видео."""
        try:
            self.log("Начинаю процесс раскадровки...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.log("Ошибка: Не удалось открыть видеофайл.")
                raise IOError("Cannot open video file")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.log(f"Всего кадров: {total_frames}, FPS: {fps:.2f}")

            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # Конец видео
                
                # Сохраняем кадр как PNG файл
                # Нумерация с 6 знаками (000001, 000002...) важна для Stable Diffusion
                frame_filename = os.path.join(output_folder, f"frame_{count:06d}.png")
                cv2.imwrite(frame_filename, frame)
                
                count += 1
                
                # Обновляем прогресс-бар (это нужно делать в основном потоке)
                progress = count / total_frames
                self.after(0, lambda p=progress: self.progress_bar.set(p))

                if count % 100 == 0:
                    self.log(f"Обработано {count}/{total_frames} кадров...")

            self.log(f"Готово! Все {count} кадров сохранены в папку: {output_folder}")

        except Exception as e:
            self.log(f"Произошла ошибка: {e}")
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            # Включаем кнопку обратно (в основном потоке)
            self.after(0, lambda: self.start_button.configure(state="normal", text="Начать раскадровку"))


if __name__ == "__main__":
    app = App()
    app.mainloop()
