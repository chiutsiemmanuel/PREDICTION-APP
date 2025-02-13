import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QWidget, QComboBox, QTableWidget, QTableWidgetItem, QListWidget,
    QAbstractItemView, QScrollArea, QFrame, QLineEdit)

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mplcursors

from statsmodels.tsa.statespace.sarimax import SARIMAX

class DataProcessingThread(QThread):
    data_loaded = pyqtSignal(pd.DataFrame)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        if self.file_path.endswith(".csv"):
            data = pd.read_csv(self.file_path)
        elif self.file_path.endswith(".xlsx"):
            data = pd.read_excel(self.file_path)
        self.data_loaded.emit(data)

class DataApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Visualization & Prediction App")
        self.setGeometry(100, 100, 1200, 800)

        self.data = None
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.initUI()

    def initUI(self):
        # Main layout
        main_layout = QHBoxLayout()

        # Sidebar layout
        sidebar_layout = QVBoxLayout()

        # Scroll area for sidebar
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)

        # Title
        title_label = QLabel("Data Dashboard")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        scroll_layout.addWidget(title_label)

        # Upload Section
        upload_button = QPushButton("Upload CSV/Excel")
        upload_button.setStyleSheet("font-size: 16px;")
        upload_button.clicked.connect(self.load_file)
        scroll_layout.addWidget(upload_button)

        self.file_label = QLabel()
        scroll_layout.addWidget(self.file_label)

        # Table Widget
        self.table_widget = QTableWidget()
        self.table_widget.setFixedHeight(200)
        scroll_layout.addWidget(self.table_widget)

        # Column Selection
        column_label = QLabel("Select Columns")
        scroll_layout.addWidget(column_label)

        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QAbstractItemView.MultiSelection)
        scroll_layout.addWidget(self.column_list)

        # Visualization Options
        vis_label = QLabel("Select Visualization Type")
        scroll_layout.addWidget(vis_label)

        self.vis_combo = QComboBox()
        self.vis_combo.addItems(["Line Plot", "Bar Plot", "Scatter Plot", "Pie Chart"])
        scroll_layout.addWidget(self.vis_combo)

        vis_button = QPushButton("Visualize Data")
        vis_button.setStyleSheet("font-size: 16px;")
        vis_button.clicked.connect(self.visualize_data)
        scroll_layout.addWidget(vis_button)

        # Prediction Options
        pred_label = QLabel("Select Prediction Graph Type")
        scroll_layout.addWidget(pred_label)

        self.pred_combo = QComboBox()
        self.pred_combo.addItems(["Line Plot", "Bar Plot"])
        scroll_layout.addWidget(self.pred_combo)

        # Prediction Year Input
        year_label = QLabel("Enter Year to Predict:")
        self.year_input = QLineEdit(self)
        self.year_input.setPlaceholderText("e.g., 2024")
        scroll_layout.addWidget(year_label)
        scroll_layout.addWidget(self.year_input)

        pred_button = QPushButton("Predict Future Values")
        pred_button.setStyleSheet("font-size: 16px;")
        pred_button.clicked.connect(self.predict_data)
        scroll_layout.addWidget(pred_button)

        # Clear Graph Button
        clear_button = QPushButton("Clear Graphs")
        clear_button.setStyleSheet("font-size: 16px;")
        clear_button.clicked.connect(self.clear_graphs)
        scroll_layout.addWidget(clear_button)

        # Download Results
        download_button = QPushButton("Download Results")
        download_button.setStyleSheet("font-size: 16px;")
        download_button.clicked.connect(self.download_results)
        scroll_layout.addWidget(download_button)

        # Add scrollable sidebar to main layout
        scroll_area.setFixedWidth(300)
        sidebar_layout.addWidget(scroll_area)
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar_layout)
        main_layout.addWidget(scroll_area)

        # Matplotlib Canvas
        main_layout.addWidget(self.canvas)

        # Set main layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)", options=options)
        if file_path:
            self.file_label.setText(file_path)
            self.processing_thread = DataProcessingThread(file_path)
            self.processing_thread.data_loaded.connect(self.handle_data_loaded)
            self.processing_thread.start()

    def handle_data_loaded(self, data):
        self.data = data
        self.data.columns = self.data.columns.str.strip()  # Remove leading/trailing spaces from column names
        self.update_table()
        self.update_columns()

    def update_table(self):
        if self.data is not None:
            self.table_widget.setRowCount(min(100, self.data.shape[0]))  # Show only first 100 rows
            self.table_widget.setColumnCount(self.data.shape[1])
            self.table_widget.setHorizontalHeaderLabels(self.data.columns)
            for i in range(min(100, self.data.shape[0])):
                for j in range(self.data.shape[1]):
                    self.table_widget.setItem(i, j, QTableWidgetItem(str(self.data.iat[i, j])))

    def update_columns(self):
        if self.data is not None:
            self.column_list.clear()
            self.column_list.addItems(self.data.columns)

    def show_error(self, message):
        error_label = QLabel(message)
        error_label.setStyleSheet("color: red; font-size: 14px;")
        self.statusBar().addWidget(error_label)

    def visualize_data(self):
        selected_columns = [item.text() for item in self.column_list.selectedItems()]
        if self.data is not None and selected_columns:
            numeric_cols = self.data[selected_columns].select_dtypes(include=[np.number])
            if numeric_cols.empty:
                self.show_error("Selected columns do not contain numeric data for visualization.")
                return

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            vis_type = self.vis_combo.currentText()

            if vis_type == "Line Plot":
                numeric_cols.plot(ax=ax)
            elif vis_type == "Bar Plot":
                numeric_cols.plot(kind='bar', ax=ax)
            elif vis_type == "Scatter Plot":
                for col in numeric_cols:
                    ax.scatter(range(len(self.data[col])), self.data[col], label=col)
            elif vis_type == "Pie Chart" and len(selected_columns) == 1:
                self.data[selected_columns[0]].value_counts().plot.pie(ax=ax, autopct='%1.1f%%')

            ax.legend()
            ax.set_title("Visualization")
            mplcursors.cursor(ax, hover=True)
            self.canvas.draw()

    def predict_data(self):
        selected_columns = [item.text() for item in self.column_list.selectedItems()]
        if self.data is not None and len(selected_columns) == 2:
            # Check column names before proceeding
            if selected_columns[0] not in self.data.columns or selected_columns[1] not in self.data.columns:
                self.show_error(f"Invalid column selection. Available columns are: {', '.join(self.data.columns)}")
                return

            X = self.data[[selected_columns[0]]]
            y = self.data[selected_columns[1]]

            if not pd.api.types.is_numeric_dtype(X[selected_columns[0]]) or not pd.api.types.is_numeric_dtype(y):
                self.show_error("Selected columns must contain numeric data for prediction.")
                return

            # Ensure the time column is in the correct datetime format (YYYY-MM)
            if self.data[selected_columns[0]].dtype == 'O':
                self.data[selected_columns[0]] = pd.to_datetime(self.data[selected_columns[0]], format='%Y')

            try:
                pred_year = int(self.year_input.text())
                if pred_year is None:
                    self.show_error("Please enter a valid year.")
                    return

                # SARIMAX model for prediction
                model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                model_fit = model.fit(disp=False)

                # Predict for the entire year (January to December)
                future_steps = 12
                forecast = model_fit.get_forecast(steps=future_steps)
                forecast_values = forecast.predicted_mean

                # Plot the prediction for the specified year (e.g., 2024)
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                future_dates = pd.date_range(start=f"{pred_year}-01", periods=future_steps, freq='M')
                ax.plot(future_dates, forecast_values, label=f"Predicted Data for {pred_year}", color="orange")
                ax.set_title(f"Prediction for {pred_year}")
                ax.set_xlabel("Month")
                ax.set_ylabel("Value")
                ax.legend()
                mplcursors.cursor(ax, hover=True)  # Reintroduce cursor for interaction
                self.canvas.draw()

            except Exception as e:
                self.show_error(f"Prediction error: {e}")

    def clear_graphs(self):
        self.figure.clear()
        self.canvas.draw()

    def download_results(self):
        if self.data is not None:
            result_df = pd.DataFrame({
                'Date': pd.date_range(start='2024-01', periods=12, freq='M'),
                'Predicted Sales': self.figure.get_axes()[0].lines[0].get_ydata()
            })
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
            if file_path:
                result_df.to_csv(file_path, index=False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataApp()
    window.show()
    sys.exit(app.exec_())