from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QFileDialog, QMessageBox

from app.backend import TopicClusterer
from app.gui.cluster_formatter import ClusterDetailGenerator
from app.gui.sections import InputSection, ResultsDisplay, ButtonSection
from app.gui.styled_elements import StyledProgressBar, StyledLabel

class WorkerThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, clusterer, topic, context_sentences, method, n_clusters, auto_clusters):
        super().__init__()
        self.clusterer = clusterer
        self.topic = topic
        self.context_sentences = context_sentences
        self.method = method
        self.n_clusters = n_clusters
        self.auto_clusters = auto_clusters

    def run(self):
        try:
            self.progress.emit("Extracting topic segments...")
            topic_segments = self.clusterer.extract_topic_segments(
                self.topic, self.context_sentences)

            if not topic_segments:
                self.error.emit(f"No segments found for topic '{self.topic}'")
                return

            self.progress.emit("Clustering segments...")

            if self.auto_clusters:
                segments, source_files, labels, embeddings = self.clusterer.cluster_segments(
                    topic_segments, self.method, None)
            else:
                segments, source_files, labels, embeddings = self.clusterer.cluster_segments(
                    topic_segments, self.method, self.n_clusters)

            self.progress.emit("Extracting keywords...")
            cluster_keywords = self.clusterer.extract_cluster_keywords(segments, labels)

            self.progress.emit("Generating cluster explanations...")
            explanations = self.clusterer.explain_clusters(segments, labels, embeddings, cluster_keywords)

            result = {
                'segments': segments,
                'source_files': source_files,
                'labels': labels,
                'cluster_keywords': cluster_keywords,
                'explanations': explanations,
                'embeddings': embeddings
            }

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.results_display = None
        self.progress_bar = None
        self.button_section = None
        self.input_section = None
        self.clusterer = TopicClusterer()
        self.current_result = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Interactive Topic Clustering with Explanations")
        self.setGeometry(100, 100, 1400, 900)

        # Set application-wide style
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8F9FA, stop:1 #E9ECEF);
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        self.input_section = InputSection(self)
        self.button_section = ButtonSection(self)
        self.results_display = ResultsDisplay(self)

        layout.addWidget(self.input_section)
        layout.addWidget(self.button_section)

        self.progress_bar = StyledProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addWidget(self.results_display)

        self.status_bar = StyledLabel()
        self.status_bar.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E8F5E8, stop:1 #F0FDF4);
                border: 1px solid #86EFAC;
                border-radius: 6px;
                padding: 8px 12px;
                color: #166534;
                font-weight: 500;
            }
        """)
        layout.addWidget(self.status_bar)

        self.set_status("Ready")

    def on_auto_clusters_toggled(self, checked):
        self.clusters_spin.setVisible(not checked)
        self.clusters_label.setVisible(not checked)

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)
            file_count = self.clusterer.load_files(directory)
            self.set_status(f"Loaded {file_count} files from {directory}")

    def set_status(self, message):
        self.status_bar.setText(message)

    def run_clustering(self):
        if not self.validate_inputs():
            return

        self.prepare_clustering_run()

        self.worker = WorkerThread(
            self.clusterer,
            self.topic_input.text().strip(),
            self.context_spin.value(),
            self.method_combo.currentText(),
            self.clusters_spin.value(),
            self.auto_clusters_checkbox.isChecked()
        )

        self.worker.progress.connect(self.set_status)
        self.worker.finished.connect(self.on_clustering_finished)
        self.worker.error.connect(self.on_clustering_error)
        self.worker.start()

    def validate_inputs(self):
        if not self.dir_input.text():
            QMessageBox.warning(self, "Warning", "Please select an input directory first")
            return False

        if not self.topic_input.text().strip():
            QMessageBox.warning(self, "Warning", "Please enter a topic to search for")
            return False

        return True

    def prepare_clustering_run(self):
        self.progress_bar.setVisible(True)
        self.run_btn.setEnabled(False)

    def on_clustering_finished(self, result):
        self.current_result = result
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.results_display.display_results(result)
        self.set_status("Clustering completed successfully")

    def on_clustering_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", error_msg)
        self.set_status(f"Error: {error_msg}")

    def on_cluster_selected(self, row):
        if not self.current_result or row < 0:
            return

        cluster_ids = sorted(set(self.current_result['labels']))
        if row >= len(cluster_ids):
            return

        cluster_id = cluster_ids[row]
        self.show_cluster_details(cluster_id)

    def show_cluster_details(self, cluster_id):
        detail_generator = ClusterDetailGenerator(
            self.current_result,
            cluster_id,
            self.method_combo.currentText(),
            self.context_spin.value(),
            self.topic_input.text()
        )

        details = detail_generator.generate_details()
        self.cluster_details.setPlainText(details)

    def save_results(self):
        if not self.current_result:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "clusters_explained.csv", "CSV Files (*.csv)")

        if filename:
            try:
                self.clusterer.save_results(
                    self.topic_input.text(),
                    self.current_result['segments'],
                    self.current_result['source_files'],
                    self.current_result['labels'],
                    self.current_result['cluster_keywords'],
                    filename,
                    self.current_result['explanations']
                )
                self.set_status(f"Results with explanations saved to {filename}")
                QMessageBox.information(self, "Success",
                                        f"Results with detailed explanations saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")