# Patient Monitoring System

The **Patient Monitoring System** is a graphical user interface (GUI) application designed for real-time or recorded electrocardiogram (ECG) signal analysis. It enables healthcare professionals to monitor patient vitals, detect arrhythmias, and trigger alerts for abnormal heart rhythms. The system integrates feature extraction and classification techniques to identify different types of arrhythmias and provide insights into heart health.

## Features
- **Real-Time & Recorded ECG Signal Display**: Visualizes ECG waveforms from live patient data or pre-recorded datasets.
- **Arrhythmia Detection & Classification**: Automatically detects and classifies at least five types of arrhythmias:
  - Atrial Fibrillation
  - Atrial Flutter
  - Sinus Bradycardia
  - Sinus Tachycardia
  - Supraventricular Tachycardia (SVT)
  - **Normal Rhythm Detection**: Determines if no arrhythmia is present.
- **Feature Extraction**:
  - **Time Domain**: Kurtosis calculation for waveform shape analysis.
  - **Frequency Domain**: Extraction of dominant frequencies to differentiate arrhythmias.
- **Additional Vital Signs Monitoring**:
  - **SpO2 (Oxygen Saturation) Measurement**
  - **Respiratory Rate (RR) Tracking**
  - **Blood Pressure Monitoring**
  - **Body Temperature Display**
- **Automated Alerts**: Notifies healthcare staff when abnormal rhythms are detected.

## User Interface
The system's UI displays essential patient monitoring data, including:
- **ECG waveform** with detected arrhythmia classification.
- **SpO2 and respiratory rate graphs** for real-time tracking.
- **Heart rate, blood pressure, SpO2 %, respiration rate, and body temperature** displayed in a clear, color-coded format.
- **Alerts and warnings** for abnormal conditions such as low SpO2.

![Screenshot 2025-03-27 171646](https://github.com/user-attachments/assets/15a6e658-7ce6-41b5-bb31-b962672f708a)


## Try it Yourself!
1. Load a recorded ECG dataset or connect to a real-time data source.
2. The system processes the ECG signal, extracting time-domain and frequency-domain features.
3. The extracted features are used to classify arrhythmias based on predefined thresholds and machine learning models.
4. Results are displayed on the GUI, with alerts for abnormal detections.

