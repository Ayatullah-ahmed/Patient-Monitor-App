import sys
import numpy as np
from PyQt5 import  QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog
from scipy import stats, signal
import pandas as pd
from scipy.signal import find_peaks
import scipy.signal as signal

from PM_design import Ui_MainWindow

class PatientMonitor(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # Initialize timer for periodic updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_all_plots)
        self.timer.start(500)  # 1Hz update rate
        
        # Data initialization
        self.spo2_data = []
        self.spo2_displayed_values = []
        self.spo2_index = 0
        self.heart_rate = None

        # ECG data initialization
        self.file_path = None
        self.ecg_data = []
        self.time_data = []
        self.ecg_curve = self.ecgSignal.plot(pen='r')  # Red color for ECG
        self.index = 0  # Current position in the signal

        self.timer_ecg = QTimer()
        self.timer_ecg.timeout.connect(self.update_ECG_plot)

        self.rr_displayed_values = []
        self.rr_data = []
        self.rr_index = 0

        # Alarm thresholds
        self.spo2_alarm_thresholds = {
            'critical_low': 85,
            'low': 90,
            'high': 100,
            'critical_high': 105
        }

        # Connect buttons to functions
        self.laodECG.clicked.connect(self.open_file_dialog)
        self.loadSpo2.clicked.connect(self.load_spo2_data)
        self.load.clicked.connect(self.load_rr_data)

    def update_all_plots(self):
        self.update_spo2_plot()
        self.update_bp_value()
        self.update_temperature()
        self.update_respiratory_rate()
    
    def load_spo2_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Dataset", "", 
            "Text Files (*.txt);;All Files (*)", options=options
        )
        if file_name:
            try:
                with open(file_name, "r") as file:
                    self.spo2_data = [float(line.strip()) for line in file.readlines()]
                self.spo2_index = 0  # Reset index after loading new data
            except Exception as e:
                print("Error loading dataset:", e)

    def update_bp_value(self):
        systolic = np.random.randint(125, 130)
        diastolic = np.random.randint(75, 80)
        self.BPvalue.setText(f"{systolic} / {diastolic}")
            
        # Set color based on BP status
        if systolic > 140 or diastolic > 90:
            self.BPvalue.setStyleSheet("color:#FF0000;font-weight:bolder;font-size:80px;border-top:none;")
        else:
            self.BPvalue.setStyleSheet("color:#FF3B50;font-weight:bolder;font-size:80px;border-top:none;")
    
    def update_spo2_plot(self):
        max_points = 500
        if not self.spo2_data:
            return
        
        if self.spo2_index >= len(self.spo2_data):
            self.spo2_index = 0  # Restart if end is reached
        
        current_spo2 = self.spo2_data[self.spo2_index]
        self.spo2_displayed_values.append(current_spo2)
        self.spo2_index += 1
        
        if len(self.spo2_displayed_values) > max_points:
            self.spo2_displayed_values.pop(0)
        
        # Update plot ranges
        min_val = min(self.spo2_displayed_values) - 1
        max_val = max(self.spo2_displayed_values) + 1
        x_min = max(len(self.spo2_displayed_values) - 500, 0)
        x_max = max(len(self.spo2_displayed_values), 500)
        
        self.spo2Signal.setYRange(min_val, max_val) 
        self.spo2Signal.setXRange(x_min, x_max)
        self.spo2Signal.setLimits(xMin=x_min, xMax=x_max, yMin=min_val, yMax=max_val)
        
        # Update SpO2 value display
        self.spo2Value.setText(f"{current_spo2}")
        
        # Check for alarms and update display
        self.check_spo2_alarms(current_spo2)
        
        # Update plot
        self.spo2_curve.setData(self.spo2_displayed_values)
    
    def check_spo2_alarms(self, spo2_value):
        """Check SpO2 value against thresholds and display appropriate alarm"""
        if spo2_value < self.spo2_alarm_thresholds['critical_low']:
            alarm_text = "CRITICAL ALARM: SpO2 extremely low!"
            alarm_color = "#FF0000"  # Red
        elif spo2_value < self.spo2_alarm_thresholds['low']:
            alarm_text = "WARNING: SpO2 low"
            alarm_color = "#FFA500"  # Orange
        elif spo2_value > self.spo2_alarm_thresholds['critical_high']:
            alarm_text = "CRITICAL ALARM: SpO2 extremely high!"
            alarm_color = "#FF0000"  # Red
        elif spo2_value > self.spo2_alarm_thresholds['high']:
            alarm_text = "WARNING: SpO2 high"
            alarm_color = "#FFA500"  # Orange
        else:
            alarm_text = "Normal"
            alarm_color = "#00FF00"  # Green
        
        # Update alarm display
        self.spo2Alarm.setText(alarm_text)
        self.spo2Alarm.setStyleSheet(
            f"color:{alarm_color};border:none;font-weight:bolder;font-size:20px;"
            "padding-left:20px;border-bottom:1px solid white;"
        )
        
        # Also update the value color to match alarm status
        self.spo2Value.setStyleSheet(
            "color:#00B7E2;font-weight:bolder;font-size:80px;border-top:none;"
        )

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Open ECG File", "", 
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)", options=options
        )
        if file_path:
            self.file_path = file_path
            self.load_ECG_data()

    def load_ECG_data(self):
        """Load ECG data from a CSV file with headers."""
        if not self.file_path:
            print("Error: No file selected.")
            return
        try:
            df = pd.read_csv(self.file_path)  # Load CSV into Pandas DataFrame
            self.ecg_data = df['signal'].values  # Extract ECG signal column
            if 'Time' not in df.columns:
                step_value = 0.001   # in seconds
                self.time_data = np.arange(
                    start=0, 
                    stop=step_value * len(self.ecg_data), 
                    step=step_value
                )  
            else:
                self.time_data = df['Time'].values  # Extract time column
            self.index = 0  # Reset position
            self.extract_features(self.file_path)
            self.timer_ecg.start(5)  # Start updating every 5ms

        except Exception as e:
            print("Error loading ECG file:", e)

    def extract_features(self, file_path):
        try:
            fs = 1 / (self.time_data[1] - self.time_data[0])  # Sampling rate

            # Frequency-Domain Features
            freqs, psd = signal.welch(self.ecg_data, fs, nperseg=1024)
            spectral_entropy = stats.entropy(psd + 1e-10)
            dominant_freq = freqs[np.argmax(psd)]
            
            # Time-Domain Features
            kurtosis = stats.kurtosis(self.ecg_data)

            # Arrhythmia Classification
            classification = self.classify_arrhythmia(dominant_freq, kurtosis)
            arrhythmia_result = (classification)
            
            # Display Features and Arrhythmia Classification
            features_str = f"{arrhythmia_result}"
            alarm_color = "#00FF00"  # Green
            self.arrythmiaType.setText(f"Arrhythmia Detected: {arrhythmia_result}")
            self.arrythmiaType.setStyleSheet(
                f"color:{alarm_color};border:none;font-weight:bolder;font-size:20px;"
                "padding-left:20px;border-bottom:1px solid white;"
            )

        except Exception as e:
            pass

    def update_ECG_plot(self):
        """Update the ECG signal dynamically with instant heart rate feedback"""
        if not hasattr(self, 'ecg_data') or len(self.ecg_data) == 0:
            return

        # Calculate sampling rate if not already set
        if not hasattr(self, 'fs'):
            if len(self.time_data) > 1:
                self.fs = 1 / (self.time_data[1] - self.time_data[0])
            else:
                self.fs = 250  # default sampling rate

        display_window = 500  # Number of points to display at a time

        # Define the range of points to display
        start_index = max(0, self.index - display_window + 1)
        x = np.arange(start_index, self.index + 1)  # Shift x-axis dynamically
        y = self.ecg_data[start_index:self.index + 1]

        self.ecg_curve.setData(x, y)

        # Calculate heart rate on sliding window (last 2 seconds of data)
        window_size = int(2 * self.fs)  # 2 second window
        if self.index >= window_size:
            current_window = self.ecg_data[self.index - window_size:self.index]
            heart_rate = self.calculate_heart_rate(current_window, self.fs)
            
            # Display heart rate immediately when first calculation is available
            if heart_rate is not None and heart_rate > 0:
                self.HRvalue.setText(f"{heart_rate:.0f}")
                # Color coding
                if heart_rate > 100:
                    self.HRvalue.setStyleSheet(
                        "color: red; font-size: 85px;border-top:none;"
                    )
                elif heart_rate < 60:
                    self.HRvalue.setStyleSheet(
                        "color: yellow; font-size: 85px;border-top:none;"
                    )
                else:
                    self.HRvalue.setStyleSheet(
                        "color: #39FF5E; font-size: 85px;border-top:none;"
                    )
            else:
                self.HRvalue.setText("--")  # Show placeholder when no valid HR
        
        # For first 2 seconds, show "Calculating..." message
        elif self.index > 0:
            self.HRvalue.setText("Calculating...")
            self.HRvalue.setStyleSheet(
                "color: white; font-size: 24px;border-top:none;"
            )

        self.index += 1

        # Stop when we reach the end
        if self.index >= len(self.ecg_data):
            self.timer_ecg.stop()

    def calculate_heart_rate(self, ecg_signal, time_data):
        """Calculate heart rate (BPM) from an ECG signal."""
        if len(ecg_signal) < 2:
            return None  # Not enough data to process

        # Sampling rate calculation
        fs = 1 / (self.time_data[1] - self.time_data[0])  # Sampling frequency (Hz)
        distance = max(1, int(fs*0.6))
        # Detect R-peaks using find_peaks
        peaks, _ = signal.find_peaks(
            ecg_signal, 
            height=np.mean(ecg_signal) + np.std(ecg_signal), 
            distance=distance
        )  

        if len(peaks) < 2:
            return self.heart_rate  # Not enough peaks to determine heart rate

        # Calculate RR intervals (time between consecutive R-peaks)
        rr_intervals = np.diff(self.time_data[peaks])  # Time differences between R-peaks

        # Convert RR intervals to heart rate (BPM)
        heart_rates = 60 / rr_intervals  # BPM calculation for each interval
        avg_heart_rate = np.mean(heart_rates)  # Average HR

        self.heart_rate = avg_heart_rate

        return avg_heart_rate

    def classify_arrhythmia(self, dominant_freq, kurtosis): 
        classification = None
        if kurtosis < 5 and 5 <= dominant_freq < 8:
            classification = "Atrial Fibrillation"
        elif 5 <= kurtosis <= 10 and dominant_freq>10:
            classification = "Atrial Flutter"
        elif 10 <= kurtosis <= 14 and dominant_freq<10:
            classification = "Supraventricular Tachycardia"
        elif 14 <= kurtosis < 19 and dominant_freq<2:
            classification = "Sinus Bradycardia"
        elif kurtosis >=19 and dominant_freq>10:
            classification = "Sinus Tachycardia"
        else: 
            classification = "Normal sinus rhythm, No arrhythmia"

        return classification

    def update_temperature(self):
        """
        Generate a random temperature with slight variations
        Typical normal body temperature range of 36.5-37.5°C
        """
        base_temp = 37.0  # Normal body temperature
        variation = np.random.uniform(-0.5, 0.5)  # Small random variation
        current_temp = round(base_temp + variation, 1)

        # Update temperature display
        self.tempValue.setText(str(current_temp))

        # Color code temperature
        if current_temp > 38.0 or current_temp < 36.0:
            self.tempValue.setStyleSheet(
                "color:#FF0000;font-weight:bolder;font-size:80px;border-top:none;"
            )
        else:
            self.tempValue.setStyleSheet(
                "color:#39FF5E;font-weight:bolder;font-size:80px;border-top:none;"
            )

    def load_rr_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Dataset", "", 
            "CSV Files (*.csv);;All Files (*)", options=options
        )
        if file_name:
            try:
                # Read the dataset using pandas
                df = pd.read_csv(file_name, encoding="utf-8-sig")  # Handles BOM if present

                # Assuming the RR data is in a specific column (e.g., 'rpm')
                column_name = "rpm"  # Change to the correct column in your dataset
                if column_name in df.columns:
                    self.rr_data = df[column_name].dropna().astype(float).tolist()
                else:
                    print(f"Error: Column '{column_name}' not found in dataset")
                    return

                self.rr_index = 0  # Reset index after loading new data

            except Exception as e:
                print("Error loading dataset:", e)

    def update_respiratory_rate(self):
        max_points = 500  # Maximum points to display

        if not hasattr(self, 'rr_data') or not self.rr_data:
            return

        if not hasattr(self, 'rr_displayed_values'):
            self.rr_displayed_values = []  # Initialize list for plotting

        # Restart index if reaching the end of the dataset
        if self.rr_index >= len(self.rr_data):
            self.rr_index = 0

        # Get current RR value from dataset
        current_rate = self.rr_data[self.rr_index]
        self.rr_index += 1  # Move to next data point

        # Update respiratory rate display
        self.RRvalue.setText(str(current_rate))

        # Color code respiratory rate based on value
        if current_rate < 10 or current_rate > 25:
            self.RRvalue.setStyleSheet(
                "color:#FF0000;font-weight:bolder;font-size:80px;border-top:none;"
            )
        else:
            self.RRvalue.setStyleSheet(
                "color:#E8D34B;font-weight:bolder;font-size:80px;border-top:none;"
            )

        # === Update RR Plot ===
        self.rr_displayed_values.append(current_rate)  # Add new value

        # Keep only the latest `max_points` values
        if len(self.rr_displayed_values) > max_points:
            self.rr_displayed_values.pop(0)

        # Define the Y-axis and X-axis range
        min_val = min(self.rr_displayed_values) - 1
        max_val = max(self.rr_displayed_values) + 1
        x_min = max(len(self.rr_displayed_values) - max_points, 0)
        x_max = max(len(self.rr_displayed_values), max_points)

        # Update the plot axes
        self.rrSignal.setYRange(min_val, max_val)
        self.rrSignal.setXRange(x_min, x_max)
        self.rrSignal.setLimits(xMin=x_min, xMax=x_max, yMin=min_val, yMax=max_val)

        # Update RR curve with new data
        self.rr_curve.setData(self.rr_displayed_values)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    monitor = PatientMonitor()
    monitor.show()
    sys.exit(app.exec_())