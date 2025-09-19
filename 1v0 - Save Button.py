import sys
import io
import torch
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QTabWidget, QLineEdit, QPushButton, QFileDialog
)
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
from PIL import Image, ImageOps

# --- Hugging Face Libraries ---
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting
from diffusers.utils import load_image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ANALYSIS_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
TEXT_TO_IMAGE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
INPAINTING_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {DEVICE}")

# ==============================================================================
# STYLESHEET FOR A DARK THEME
# ==============================================================================
DARK_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #f0f0f0; font-size: 14px; }
QTabWidget::pane { border: 1px solid #444; }
QTabBar::tab { background: #3c3c3c; color: #f0f0f0; padding: 10px; border: 1px solid #444; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background: #505050; margin-bottom: -1px; }
QLabel { border: 2px dashed #555; border-radius: 10px; font-size: 16px; color: #888; }
QLineEdit, QTextEdit { background-color: #3c3c3c; border: 1px solid #555; border-radius: 4px; padding: 5px; }
QPushButton { background-color: #007acc; color: white; border: none; padding: 10px; border-radius: 4px; }
QPushButton:hover { background-color: #008ae6; }
QPushButton:pressed { background-color: #006bb3; }
"""

# ==============================================================================
# AI WORKER THREAD
# ==============================================================================
class AIWorker(QObject):
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, task_function, **kwargs):
        super().__init__()
        self.task_function = task_function
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.task_function(**self.kwargs)
            self.result_ready.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
class AIStudioApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Unified AI Studio')
        self.resize(800, 850)
        self.setAcceptDrops(True)

        self.analyzer_processor = None
        self.analyzer_model = None
        self.txt2img_pipeline = None
        self.inpainting_pipeline = None
        self.current_image_path = None
        self.last_txt2img_result = None
        self.last_expansion_result = None

        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.create_analyzer_tab()
        self.create_txt2img_tab()
        self.create_expansion_tab()
        
    def get_analyzer_components(self):
        if self.analyzer_processor is None or self.analyzer_model is None:
            self.update_status_on_tab(0, "Loading LLaVA Model & Processor...")
            QApplication.processEvents()
            self.analyzer_processor = AutoProcessor.from_pretrained(ANALYSIS_MODEL_ID)
            self.analyzer_model = LlavaForConditionalGeneration.from_pretrained(
                ANALYSIS_MODEL_ID, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True
            ).to(DEVICE)
        return self.analyzer_processor, self.analyzer_model

    def get_txt2img_pipeline(self):
        if self.txt2img_pipeline is None:
            self.update_status_on_tab(1, "Loading Text-to-Image Model...")
            QApplication.processEvents()
            self.txt2img_pipeline = AutoPipelineForText2Image.from_pretrained(
                TEXT_TO_IMAGE_MODEL_ID, torch_dtype=TORCH_DTYPE, variant="fp16", use_safetensors=True
            ).to(DEVICE)
        return self.txt2img_pipeline

    def get_inpainting_pipeline(self):
        if self.inpainting_pipeline is None:
            self.update_status_on_tab(2, "Loading Image Expansion Model...")
            QApplication.processEvents()
            self.inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
                INPAINTING_MODEL_ID, torch_dtype=TORCH_DTYPE, variant="fp16", use_safetensors=True
            ).to(DEVICE)
        return self.inpainting_pipeline

    def create_analyzer_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.analyzer_image_label = QLabel('Drag & Drop Image for Analysis')
        self.analyzer_result_text = QTextEdit()
        self.analyzer_result_text.setReadOnly(True)
        layout.addWidget(self.analyzer_image_label, 1)
        layout.addWidget(self.analyzer_result_text, 1)
        self.tabs.addTab(tab, "1. Image Analyzer")

    def create_txt2img_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        prompt_layout = QHBoxLayout()
        self.txt2img_prompt = QLineEdit()
        self.txt2img_prompt.setPlaceholderText("e.g., a photorealistic astronaut riding a horse on Mars")
        self.txt2img_button = QPushButton("Generate")
        self.txt2img_save_button = QPushButton("Save Image")
        self.txt2img_save_button.hide()
        prompt_layout.addWidget(self.txt2img_prompt)
        prompt_layout.addWidget(self.txt2img_button)
        prompt_layout.addWidget(self.txt2img_save_button)
        self.txt2img_result_label = QLabel('Generated image will appear here')
        layout.addLayout(prompt_layout)
        layout.addWidget(self.txt2img_result_label, 1)
        self.tabs.addTab(tab, "2. Text to Image")
        self.txt2img_button.clicked.connect(self.run_txt2img_task)
        self.txt2img_save_button.clicked.connect(lambda: self.save_image(self.last_txt2img_result))

    def create_expansion_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.expansion_image_label = QLabel('1. Drop initial image here')
        prompt_layout = QHBoxLayout()
        self.expansion_prompt = QLineEdit()
        self.expansion_prompt.setPlaceholderText("2. Describe what to add (e.g., 'full body, wearing blue jeans')")
        self.expansion_button = QPushButton("Expand")
        self.expansion_save_button = QPushButton("Save Image")
        self.expansion_save_button.hide()
        prompt_layout.addWidget(self.expansion_prompt)
        prompt_layout.addWidget(self.expansion_button)
        prompt_layout.addWidget(self.expansion_save_button)
        self.expansion_result_label = QLabel('Expanded image will appear here')
        layout.addWidget(self.expansion_image_label, 1)
        layout.addLayout(prompt_layout)
        layout.addWidget(self.expansion_result_label, 1)
        self.tabs.addTab(tab, "3. Image Expansion")
        self.expansion_button.clicked.connect(self.run_expansion_task)
        self.expansion_save_button.clicked.connect(lambda: self.save_image(self.last_expansion_result))

    def save_image(self, image_to_save):
        if not image_to_save: return
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image As...", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)")
        if filePath:
            try:
                image_to_save.save(filePath)
            except Exception as e:
                print(f"Error saving file: {e}")

    def start_worker(self, task_function, on_result, on_error, **kwargs):
        self.thread = QThread()
        self.worker = AIWorker(task_function, **kwargs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.worker.deleteLater)
        self.worker.result_ready.connect(on_result)
        self.worker.error.connect(on_error)
        self.thread.start()

    def run_analysis_task(self, image_path):
        def task(path):
            processor, model = self.get_analyzer_components()
            image = Image.open(path)
            prompt = "USER: <image>\nWhat are the objects, people, and actions in this image? Describe it in detail.\nASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, TORCH_DTYPE)
            output_ids = model.generate(**inputs, max_new_tokens=512)
            input_token_len = inputs.input_ids.shape[1]
            decoded_text = processor.decode(output_ids[0][input_token_len:], skip_special_tokens=True)
            return decoded_text.strip()
        self.update_status_on_tab(0, "Analyzing image...")
        self.start_worker(task, self.on_analysis_complete, self.on_task_error, path=image_path)

    def run_txt2img_task(self):
        self.txt2img_save_button.hide()
        prompt = self.txt2img_prompt.text()
        if not prompt: return
        def task(p):
            pipe = self.get_txt2img_pipeline()
            image = pipe(prompt=p, num_inference_steps=25, width=768, height=768).images[0]
            return image
        self.update_status_on_tab(1, f"Generating: '{prompt[:40]}...'")
        self.start_worker(task, self.on_generation_complete, self.on_task_error, p=prompt)

    def run_expansion_task(self):
        self.expansion_save_button.hide()
        prompt = self.expansion_prompt.text()
        if not self.current_image_path or not prompt: return
        def task(path, p):
            pipe = self.get_inpainting_pipeline()
            init_image = load_image(path).resize((1024, 1024))
            mask_image = Image.new("RGB", (1024, 1024), "black")
            mask_image.paste(Image.new("RGB", (1024, 512), "white"), (0, 512))
            final_image = Image.new("RGB", (1024, 1024), "black")
            final_image.paste(load_image(path).resize((1024, 512)), (0, 0))
            image = pipe(prompt=p, image=final_image, mask_image=mask_image, strength=0.9, num_inference_steps=30).images[0]
            return image
        self.update_status_on_tab(2, "Expanding image...")
        self.start_worker(task, self.on_expansion_complete, self.on_task_error, path=self.current_image_path, p=prompt)

    def on_analysis_complete(self, result):
        self.analyzer_result_text.setText(result)
        
    def on_generation_complete(self, result_image):
        self.display_image(self.txt2img_result_label, result_image)
        self.last_txt2img_result = result_image
        self.txt2img_save_button.show()
        
    def on_expansion_complete(self, result_image):
        self.display_image(self.expansion_result_label, result_image)
        self.last_expansion_result = result_image
        self.expansion_save_button.show()

    def on_task_error(self, error_message):
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index == 0: self.analyzer_result_text.setText(f"Error:\n{error_message}")
        elif current_tab_index == 1: self.txt2img_result_label.setText(f"Error: {error_message}")
        elif current_tab_index == 2: self.expansion_result_label.setText(f"Error: {error_message}")

    def update_status_on_tab(self, tab_index, message):
        label = self.tabs.widget(tab_index).findChild(QLabel)
        if label: label.setText(message)

    def display_image(self, label, pil_image):
        with io.BytesIO() as buffer:
            pil_image.save(buffer, 'PNG')
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls: return
        file_path = urls[0].toLocalFile()
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')): return
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index == 0:
            self.display_image(self.analyzer_image_label, Image.open(file_path))
            self.run_analysis_task(file_path)
        elif current_tab_index == 2:
            self.current_image_path = file_path
            self.display_image(self.expansion_image_label, Image.open(file_path))
            self.expansion_result_label.setText("Image loaded. Now enter prompt and expand.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    main_window = AIStudioApp()
    main_window.show()
    sys.exit(app.exec())