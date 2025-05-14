import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QSlider, QToolBar,
                            QAction, QFileDialog, QProgressBar, QStyle, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Application")
        self.setGeometry(0, 0, 1920, 1080)
        self.showMaximized()
        
        # Initialize variables
        self.original_image = None
        self.processed_image = None
        self.image_history = []
        self.current_history_index = -1
        self.is_live_preview = False
        self.cap = None
        self.is_drawing = False
        self.last_point = None
        self.drawing_points = []
        self.drawing_color = (0, 255, 0)  # Green color for drawing
        self.drawing_thickness = 2
        
        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)
        
        # Add title label
        self.title_label = QLabel("<h1 style='color:#0078d7; letter-spacing:2px;'>Sleek Image Processing App</h1>")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("margin-bottom: 10px; text-shadow: 1px 1px 2px #aaa;")
        self.layout.addWidget(self.title_label)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet('''
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f0f4ff, stop:1 #dbeafe);
            border-radius: 24px;
            border: 2px solid #0078d7;
            margin: 10px;
            box-shadow: 0px 4px 24px #0078d733;
        ''')
        self.image_label.setMinimumSize(1200, 900)
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        self.layout.addWidget(self.image_label, stretch=1)
        
        # Create control panel
        self.create_control_panel()
        
        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(30)
        self.progress_bar.setStyleSheet("font-size: 16px; padding: 5px;")
        self.layout.addWidget(self.progress_bar)
        
        # Set up drag and drop
        self.setAcceptDrops(True)
        
        # Set initial theme
        self.set_theme("light")
        
    def create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setStyleSheet('''
            QToolBar {
                background: #e0e7ff;
                border-bottom: 2px solid #0078d7;
                padding: 8px;
            }
            QToolButton {
                font-size: 16px;
                color: #22223b;
                background: #c7d2fe;
                border-radius: 8px;
                margin: 2px;
                padding: 6px 12px;
            }
            QToolButton:hover {
                background: #a5b4fc;
                color: #0078d7;
            }
        ''')
        self.addToolBar(toolbar)
        
        # Add toolbar actions with icons
        open_action = QAction("Open Image", self)
        open_action.setToolTip("Open an image file")
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)
        
        save_action = QAction("Save Image", self)
        save_action.setToolTip("Save the processed image")
        save_action.triggered.connect(self.save_image)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # Add filter actions
        filters = ["Grayscale", "Gaussian Blur", "Edge Detection", "Sharpen", 
                  "Brightness/Contrast", "Hue Adjustment"]
        for filter_name in filters:
            action = QAction(filter_name, self)
            action.setToolTip(f"Apply {filter_name} filter")
            action.triggered.connect(lambda checked, f=filter_name: self.apply_filter(f))
            toolbar.addAction(action)
        
        toolbar.addSeparator()
        
        # Undo/Redo actions
        undo_action = QAction("Undo", self)
        undo_action.setToolTip("Undo last action")
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.setToolTip("Redo last undone action")
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)
        
        toolbar.addSeparator()
        
        # Theme toggle
        theme_action = QAction("Toggle Theme", self)
        theme_action.setToolTip("Switch between light and dark mode")
        theme_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(theme_action)
        
    def create_control_panel(self):
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_panel.setStyleSheet('''
            background: #f1f5f9;
            border-radius: 16px;
            margin-top: 10px;
        ''')
        
        # Add control buttons
        buttons = [
            ("Open", self.open_image),
            ("Grayscale", lambda: self.apply_filter("Grayscale")),
            ("Gaussian Blur", lambda: self.apply_filter("Gaussian Blur")),
            ("Edge Detection", lambda: self.apply_filter("Edge Detection")),
            ("Sharpen", lambda: self.apply_filter("Sharpen")),
            ("Reset", self.reset_image),
            ("Save", self.save_image),
            ("Start Live", self.start_live_preview),
            ("Stop Live", self.stop_live_preview),
            ("Clear Drawing", self.clear_drawing)
        ]
        
        for text, callback in buttons:
            btn = QPushButton(text)
            btn.setToolTip(text)
            btn.setStyleSheet('''
                QPushButton {
                    background: #6366f1;
                    color: #fff;
                    border-radius: 8px;
                    font-size: 15px;
                    padding: 8px 18px;
                    margin: 2px;
                }
                QPushButton:hover {
                    background: #818cf8;
                    color: #22223b;
                }
            ''')
            btn.clicked.connect(callback)
            control_layout.addWidget(btn)
        
        # Add sliders
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(1, 15)
        self.blur_slider.setValue(5)
        self.blur_slider.valueChanged.connect(lambda: self.apply_filter("Gaussian Blur"))
        self.blur_slider.setToolTip("Adjust Gaussian Blur intensity (odd values only)")
        self.blur_slider.setStyleSheet('''
            QSlider::groove:horizontal {
                border: 1px solid #6366f1;
                height: 8px;
                background: #e0e7ff;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6366f1;
                border: 1px solid #818cf8;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        ''')
        control_layout.addWidget(QLabel("Blur:"))
        control_layout.addWidget(self.blur_slider)
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(lambda: self.apply_filter("Brightness/Contrast"))
        self.brightness_slider.setToolTip("Adjust brightness (-100 to 100)")
        self.brightness_slider.setStyleSheet('''
            QSlider::groove:horizontal {
                border: 1px solid #6366f1;
                height: 8px;
                background: #e0e7ff;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6366f1;
                border: 1px solid #818cf8;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        ''')
        control_layout.addWidget(QLabel("Brightness:"))
        control_layout.addWidget(self.brightness_slider)
        
        self.layout.addWidget(control_panel)
        
    def open_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                     "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_name:
                self.original_image = cv2.imread(file_name)
                if self.original_image is None:
                    raise Exception("Failed to load image")
                self.processed_image = self.original_image.copy()
                self.update_image_display()
                self.add_to_history()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open image: {str(e)}")
            
    def save_image(self):
        try:
            if self.processed_image is not None:
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", 
                                                         "Image Files (*.png *.jpg *.jpeg *.bmp)")
                if file_name:
                    success = cv2.imwrite(file_name, self.processed_image)
                    if not success:
                        raise Exception("Failed to save image")
                    QMessageBox.information(self, "Success", "Image saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
                
    def apply_filter(self, filter_name):
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "Please open an image first!")
            return
            
        try:
            if filter_name == "Grayscale":
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
            elif filter_name == "Gaussian Blur":
                kernel_size = self.blur_slider.value()
                # Ensure kernel size is a positive odd integer
                if kernel_size < 1:
                    kernel_size = 1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                self.processed_image = cv2.GaussianBlur(self.processed_image, (kernel_size, kernel_size), 0)
            elif filter_name == "Edge Detection":
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            elif filter_name == "Sharpen":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                self.processed_image = cv2.filter2D(self.processed_image, -1, kernel)
            elif filter_name == "Brightness/Contrast":
                brightness = self.brightness_slider.value()
                self.processed_image = cv2.convertScaleAbs(self.processed_image, alpha=1, beta=brightness)
            elif filter_name == "Hue Adjustment":
                hsv = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2HSV)
                hsv[:,:,0] = (hsv[:,:,0] + 30) % 180
                self.processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
            self.update_image_display()
            self.add_to_history()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply filter: {str(e)}")
        
    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.update_image_display()
            self.add_to_history()
        else:
            QMessageBox.warning(self, "Warning", "No image to reset!")
            
    def update_image_display(self):
        if hasattr(self, 'processed_image') and self.processed_image is not None:
            img = self.processed_image
            # Always convert BGR to RGB for display
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            # Dynamically scale to label size
            label_size = self.image_label.size()
            self.image_label.setPixmap(pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
    def add_to_history(self):
        if self.processed_image is not None:
            self.image_history = self.image_history[:self.current_history_index + 1]
            self.image_history.append(self.processed_image.copy())
            self.current_history_index = len(self.image_history) - 1
            
    def undo(self):
        if self.current_history_index > 0:
            self.current_history_index -= 1
            self.processed_image = self.image_history[self.current_history_index].copy()
            self.update_image_display()
            
    def redo(self):
        if self.current_history_index < len(self.image_history) - 1:
            self.current_history_index += 1
            self.processed_image = self.image_history[self.current_history_index].copy()
            self.update_image_display()
            
    def start_live_preview(self):
        try:
            self.is_live_preview = True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_live_preview)
            self.timer.start(30)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start live preview: {str(e)}")
            self.stop_live_preview()
        
    def stop_live_preview(self):
        self.is_live_preview = False
        if self.cap is not None:
            self.cap.release()
        if hasattr(self, 'timer'):
            self.timer.stop()
            
    def update_live_preview(self):
        if self.is_live_preview and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.processed_image = frame
                self.update_image_display()
                
    def mouse_press_event(self, event):
        if self.processed_image is not None:
            self.is_drawing = True
            self.last_point = event.pos()
            
    def mouse_move_event(self, event):
        if self.is_drawing and self.processed_image is not None:
            current_point = event.pos()
            if self.last_point is not None:
                # Convert QPoint to image coordinates
                img_rect = self.get_image_rect()
                if img_rect.contains(current_point):
                    start_x = int((self.last_point.x() - img_rect.x()) * self.processed_image.shape[1] / img_rect.width())
                    start_y = int((self.last_point.y() - img_rect.y()) * self.processed_image.shape[0] / img_rect.height())
                    end_x = int((current_point.x() - img_rect.x()) * self.processed_image.shape[1] / img_rect.width())
                    end_y = int((current_point.y() - img_rect.y()) * self.processed_image.shape[0] / img_rect.height())
                    
                    cv2.line(self.processed_image, (start_x, start_y), (end_x, end_y), 
                            self.drawing_color, self.drawing_thickness)
                    self.update_image_display()
            self.last_point = current_point
            
    def mouse_release_event(self, event):
        self.is_drawing = False
        self.last_point = None
        if self.processed_image is not None:
            self.add_to_history()
            
    def get_image_rect(self):
        if self.processed_image is None:
            return None
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return None
        return self.image_label.rect()
        
    def clear_drawing(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.update_image_display()
            self.add_to_history()
            
    def set_theme(self, theme):
        if theme == "dark":
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #3b3b3b;
                    color: #ffffff;
                    border: 1px solid #555555;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #4b4b4b;
                }
                QLabel {
                    color: #ffffff;
                }
                QSlider::groove:horizontal {
                    border: 1px solid #999999;
                    height: 8px;
                    background: #4b4b4b;
                    margin: 2px 0;
                }
                QSlider::handle:horizontal {
                    background: #ffffff;
                    border: 1px solid #5c5c5c;
                    width: 18px;
                    margin: -2px 0;
                    border-radius: 3px;
                }
            """)
        else:
            self.setStyleSheet("")
            
    def toggle_theme(self):
        current_theme = "dark" if self.styleSheet() else "light"
        self.set_theme("light" if current_theme == "dark" else "dark")
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
            
    def dropEvent(self, event):
        try:
            files = [u.toLocalFile() for u in event.mimeData().urls()]
            if files:
                self.original_image = cv2.imread(files[0])
                if self.original_image is None:
                    raise Exception("Failed to load image")
                self.processed_image = self.original_image.copy()
                self.update_image_display()
                self.add_to_history()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dropped image: {str(e)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'processed_image'):
            self.update_image_display()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
