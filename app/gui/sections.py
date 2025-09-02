from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTextEdit, QFileDialog, QHBoxLayout, QTableWidget, \
    QTableWidgetItem, QHeaderView, QSplitter, QLabel, QListWidget

from app.gui.cluster_formatter import ClusterResultFormatter
from app.gui.styled_elements import StyledGroupBox, StyledLineEdit, StyledLabel, StyledButton, StyledSpinBox, \
    StyledComboBox, StyledCheckBox, StyledTabWidget, StyledTextEdit


class InputSection(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        input_group = StyledGroupBox("Input Settings")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(15)

        input_layout.addLayout(self.create_directory_section())
        input_layout.addLayout(self.create_topic_section())
        input_layout.addLayout(self.create_parameters_section())

        input_group.setLayout(input_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(input_group)

    def create_directory_section(self):
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(12)
        dir_layout.addWidget(StyledLabel("Input Directory:"))
        self.parent.dir_input = StyledLineEdit()
        self.parent.dir_input.setPlaceholderText("Select directory containing text files")
        dir_layout.addWidget(self.parent.dir_input)
        self.parent.browse_btn = StyledButton("Browse")
        self.parent.browse_btn.clicked.connect(self.parent.browse_directory)
        dir_layout.addWidget(self.parent.browse_btn)
        return dir_layout

    def create_topic_section(self):
        topic_layout = QHBoxLayout()
        topic_layout.setSpacing(12)
        topic_layout.addWidget(StyledLabel("Topic:"))
        self.parent.topic_input = StyledLineEdit()
        self.parent.topic_input.setPlaceholderText("Enter topic to search for (e.g., AYA, pediatrie)")
        topic_layout.addWidget(self.parent.topic_input)
        return topic_layout

    def create_parameters_section(self):
        params_layout = QHBoxLayout()
        params_layout.setSpacing(15)

        params_layout.addWidget(StyledLabel("Context Sentences:"))
        self.parent.context_spin = StyledSpinBox()
        self.parent.context_spin.setRange(1, 10)
        self.parent.context_spin.setValue(3)
        params_layout.addWidget(self.parent.context_spin)

        params_layout.addWidget(StyledLabel("Clustering Method:"))
        self.parent.method_combo = StyledComboBox()
        self.parent.method_combo.addItems(["kmeans", "hdbscan"])
        params_layout.addWidget(self.parent.method_combo)

        self.parent.auto_clusters_checkbox = StyledCheckBox("Auto-detect clusters")
        self.parent.auto_clusters_checkbox.toggled.connect(self.parent.on_auto_clusters_toggled)
        params_layout.addWidget(self.parent.auto_clusters_checkbox)

        self.parent.clusters_label = StyledLabel("Number of Clusters:")
        params_layout.addWidget(self.parent.clusters_label)
        self.parent.clusters_spin = StyledSpinBox()
        self.parent.clusters_spin.setRange(1, 20)
        self.parent.clusters_spin.setValue(5)
        params_layout.addWidget(self.parent.clusters_spin)

        return params_layout


class ButtonSection(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        self.parent.run_btn = StyledButton("Run Clustering", primary=True)
        self.parent.run_btn.clicked.connect(self.parent.run_clustering)
        btn_layout.addWidget(self.parent.run_btn)

        self.parent.save_btn = StyledButton("Save Results")
        self.parent.save_btn.clicked.connect(self.parent.save_results)
        self.parent.save_btn.setEnabled(False)
        btn_layout.addWidget(self.parent.save_btn)

        btn_layout.addStretch()

        self.setLayout(btn_layout)


class ResultsDisplay(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.parent.tabs = StyledTabWidget()
        layout.addWidget(self.parent.tabs)

    def display_results(self, result):
        self.parent.tabs.clear()
        formatter = ClusterResultFormatter(result, self.parent.topic_input.text(),
                                           self.parent.method_combo.currentText(),
                                           self.parent.context_spin.value())

        self.create_summary_tab(formatter)
        self.create_explanations_tab(formatter)
        self.create_details_tab(result)
        self.create_explorer_tab(result)

    def create_summary_tab(self, formatter):
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(15, 15, 15, 15)

        summary_text = StyledTextEdit()
        summary_text.setReadOnly(True)
        summary_text.setFont(QFont("Consolas", 10))
        summary_text.setPlainText(formatter.generate_summary_report())

        summary_layout.addWidget(summary_text)
        self.parent.tabs.addTab(summary_tab, "Analysis Report")

    def create_explanations_tab(self, formatter):
        explanations_tab = QWidget()
        explanations_layout = QVBoxLayout(explanations_tab)
        explanations_layout.setContentsMargins(15, 15, 15, 15)

        explanations_text = StyledTextEdit()
        explanations_text.setReadOnly(True)
        explanations_text.setFont(QFont("Arial", 11))
        explanations_text.setPlainText(formatter.generate_explanations_report())

        explanations_layout.addWidget(explanations_text)
        self.parent.tabs.addTab(explanations_tab, "Cluster Reasoning")

    def create_details_tab(self, result):
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)

        table = self.create_results_table(result)
        details_layout.addWidget(table)
        self.parent.tabs.addTab(details_tab, "Detailed Table")

    def create_results_table(self, result):
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "Cluster", "Theme", "Coherence", "Source File", "Keywords", "Preview"
        ])

        table.setRowCount(len(result['segments']))
        for i, (segment, source_file, cluster_id) in enumerate(
                zip(result['segments'], result['source_files'], result['labels'])):
            explanation = result['explanations'].get(cluster_id, {})
            keywords = ", ".join(result['cluster_keywords'].get(cluster_id, []))
            preview = segment[:150] + "..." if len(segment) > 150 else segment

            table.setItem(i, 0, QTableWidgetItem(str(cluster_id)))
            table.setItem(i, 1, QTableWidgetItem(explanation.get('theme', '')))
            table.setItem(i, 2, QTableWidgetItem(f"{explanation.get('coherence_score', 0):.3f}"))
            table.setItem(i, 3, QTableWidgetItem(source_file))
            table.setItem(i, 4, QTableWidgetItem(keywords))
            table.setItem(i, 5, QTableWidgetItem(preview))

        header = table.horizontalHeader()
        for i in range(5):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)

        return table

    def create_explorer_tab(self, result):
        explorer_tab = QWidget()
        explorer_layout = QHBoxLayout(explorer_tab)

        left_panel = self.create_cluster_list_panel(result)
        right_panel = self.create_cluster_details_panel()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        explorer_layout.addWidget(splitter)
        self.parent.tabs.addTab(explorer_tab, "Cluster Explorer")

    def create_cluster_list_panel(self, result):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Select Cluster to Explore:"))

        self.parent.cluster_list = QListWidget()
        for cluster_id in sorted(set(result['labels'])):
            item_text = self.format_cluster_list_item(cluster_id, result)
            self.parent.cluster_list.addItem(item_text)

        self.parent.cluster_list.currentRowChanged.connect(self.parent.on_cluster_selected)
        left_layout.addWidget(self.parent.cluster_list)
        left_panel.setMaximumWidth(400)
        return left_panel

    def format_cluster_list_item(self, cluster_id, result):
        explanation = result['explanations'].get(cluster_id, {})
        cluster_mask = result['labels'] == cluster_id
        cluster_size = sum(cluster_mask)

        if cluster_id == -1:
            return f"Outliers ({cluster_size} segments)"
        else:
            theme = explanation.get('theme', 'Unknown')
            coherence = explanation.get('coherence_score', 0)
            return f"Cluster {cluster_id}: {theme} ({cluster_size} segments, coherence: {coherence:.2f})"

    def create_cluster_details_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.parent.cluster_details = QTextEdit()
        self.parent.cluster_details.setReadOnly(True)
        self.parent.cluster_details.setPlaceholderText("Select a cluster from the left to see detailed explanation...")
        right_layout.addWidget(self.parent.cluster_details)
        return right_panel

