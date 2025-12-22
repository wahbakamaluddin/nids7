# nids_gui.py
"""
Network Intrusion Detection System - GUI Application

This module provides a graphical user interface for the NIDS system,
using the modular 5-component architecture:

1. Packet Capturer - Captures live network traffic
2. Packet Parser - Converts packets to bidirectional flows
3. Feature Extractor - Extracts flow features
4. Feature Mapper - Maps features to ML model input format
5. Anomaly Detector - Classifies traffic using ML models

The GUI provides controls for:
- Starting/stopping packet capture
- Configuring network interface and model paths
- Viewing real-time detection logs
- Monitoring system resource usage
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
import time
import psutil
from collections import deque
from pathlib import Path
import os

from nids.helper.other.pipeline import NIDSPipeline
from nids.helper.other.constants import ROOT_DIR
from nids.anomaly_detector import DetectionResult


PATHS = {
    "Binary Model": os.path.join(ROOT_DIR, 'model/binary_classification/knn_binary.joblib'),
    "Multi-class Model": os.path.join(ROOT_DIR, "model/multi_class_classification/knn_multi_class.joblib"),
    "Scaler": os.path.join(ROOT_DIR, "model/binary_classification/robust_scaler.joblib"),
    "Output CSV": os.path.join(ROOT_DIR, "csv")
}

class NIDSGUI:
    """
    Graphical User Interface for the Network Intrusion Detection System.
    
    This class provides a Tkinter-based GUI that integrates with the 
    NIDSPipeline to provide real-time network intrusion detection with
    visual feedback.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CICFlowMeter - Network Flow Monitor")
        self.root.geometry("700x500")
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        
        # Pipeline instance
        self.pipeline: NIDSPipeline = None
        self.capturing = False
        
        # Resource monitoring values
        self.throughput_value = 0
        self.cpu_usage_value = 0
        self.memory_usage_value = 0
        self.packet_count = 0
        self.flow_count = 0
        
        # Performance optimizations
        self.log_queue = deque(maxlen=1000)
        self.last_update_time = 0
        self.update_interval = 0.1  # 100ms for UI updates
        self.resource_update_interval = 1.0  # 1 second for resource updates
        
        self.setup_gui()
        
    def setup_gui(self):
        """Set up the GUI components."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and Control Buttons Frame
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Title
        title_label = ttk.Label(
            header_frame, 
            text="Network Intrusion Detection System V2.0", 
            font=("Segoe UI", 14, "bold")
        )
        title_label.pack(side=tk.LEFT)
        
        # Control Buttons - positioned at top right
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.start_btn = ttk.Button(
            button_frame,
            text="Start Capture",
            command=self.start_capture,
            width=12
        )
        self.start_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(
            button_frame, 
            text="Stop Capture", 
            command=self.stop_capture,
            state="disabled",
            width=12
        )
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.clear_btn = ttk.Button(
            button_frame,
            text="Clear Log",
            command=self.clear_log,
            width=10
        )
        self.clear_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            button_frame, 
            text="Exit", 
            command=self.exit_app,
            width=8
        ).pack(side=tk.LEFT, padx=2)

        # Configuration Frame
        config_frame = ttk.Frame(main_frame)
        config_frame.pack(fill=tk.X, pady=5)
        
        # Network Interface
        ttk.Label(config_frame, text="Interface:").grid(row=0, column=0, sticky="w", padx=(0, 2))
        self.interface_entry = ttk.Entry(config_frame, width=12)
        self.interface_entry.insert(0, "wlp0s20f3")
        self.interface_entry.grid(row=0, column=1, sticky="w", padx=(0, 10))

        # Binary Model Path
        ttk.Label(config_frame, text="Model:").grid(row=0, column=2, sticky="w", padx=(0, 2))
        self.model_entry = ttk.Entry(config_frame, width=40)
        self.model_entry.insert(0, PATHS["Binary Model"])
        self.model_entry.grid(row=0, column=3, sticky="we", padx=(0, 10))
        
        # Reset button
        ttk.Button(
            config_frame, 
            text="Reset", 
            command=self.reset_parameters,
            width=8
        ).grid(row=0, column=10, sticky="e", padx=(0, 5))
        
        config_frame.columnconfigure(3, weight=1)
        config_frame.columnconfigure(10, weight=1)

        # System Metrics Frame
        metrics_frame = ttk.Frame(main_frame)
        metrics_frame.pack(fill=tk.X, pady=3)
        
        self.cpu_label = ttk.Label(metrics_frame, text="CPU: --%", font=("Consolas", 9))
        self.cpu_label.pack(side=tk.LEFT, padx=8)
        
        self.mem_label = ttk.Label(metrics_frame, text="Memory: --%", font=("Consolas", 9))
        self.mem_label.pack(side=tk.LEFT, padx=8)
        
        self.power_label = ttk.Label(metrics_frame, text="Power: --", font=("Consolas", 9))
        self.power_label.pack(side=tk.LEFT, padx=8)
        
        self.throughput_label = ttk.Label(metrics_frame, text="Throughput: -- pkt/s", font=("Consolas", 9))
        self.throughput_label.pack(side=tk.LEFT, padx=8)

        self.packet_count_label = ttk.Label(metrics_frame, text="Packets: --", font=("Consolas", 9))
        self.packet_count_label.pack(side=tk.LEFT, padx=8)

        self.flow_count_label = ttk.Label(metrics_frame, text="Flows: --", font=("Consolas", 9))
        self.flow_count_label.pack(side=tk.LEFT, padx=8)

        # Status bar
        self.status_bar = ttk.Label(
            main_frame, 
            text="Architecture: Packet Capturer → Packet Parser → Feature Extractor → Feature Mapper → Anomaly Detector", 
            font=("Consolas", 8), 
            foreground="gray"
        )
        self.status_bar.pack(fill=tk.X, pady=(0, 5))

        # Log widget
        log_frame = ttk.LabelFrame(main_frame, text="Detection Log", padding="3")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_widget = scrolledtext.ScrolledText(
            log_frame, 
            bg="black", 
            fg="lime", 
            insertbackground="white",
            font=("Consolas", 9),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.log_widget.pack(fill=tk.BOTH, expand=True)

        # Quick Access Buttons at bottom
        quick_button_frame = ttk.Frame(main_frame)
        quick_button_frame.pack(fill=tk.X, pady=3)
        
        ttk.Button(
            quick_button_frame,
            text="Clear Log",
            command=self.clear_log,
            width=10
        ).pack(side=tk.RIGHT, padx=2)
        
        ttk.Button(
            quick_button_frame,
            text="Copy Log",
            command=self.copy_log,
            width=10
        ).pack(side=tk.RIGHT, padx=2)
        
        ttk.Button(
            quick_button_frame,
            text="Save Log",
            command=self.save_log,
            width=10
        ).pack(side=tk.RIGHT, padx=2)

        ttk.Button(
            quick_button_frame,
            text="Statistics",
            command=self.show_statistics,
            width=10
        ).pack(side=tk.RIGHT, padx=2)

    def reset_parameters(self):
        """Reset parameters to default values."""
        self.interface_entry.delete(0, tk.END)
        self.interface_entry.insert(0, "wlp0s20f3")
        self.model_entry.delete(0, tk.END)
        self.model_entry.insert(0, PATHS["Binary Model"])
        self._update_log_widget("[*] Parameters reset to defaults\n")

    def copy_log(self):
        """Copy log contents to clipboard."""
        try:
            log_content = self.log_widget.get(1.0, tk.END)
            if log_content.strip():
                self.root.clipboard_clear()
                self.root.clipboard_append(log_content)
                self._update_log_widget("[*] Log copied to clipboard\n")
        except Exception as e:
            self._update_log_widget(f"[ERROR] Failed to copy log: {str(e)}\n")

    def save_log(self):
        """Save log contents to file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                log_content = self.log_widget.get(1.0, tk.END)
                with open(filename, 'w') as f:
                    f.write(log_content)
                self._update_log_widget(f"[*] Log saved to {filename}\n")
        except Exception as e:
            self._update_log_widget(f"[ERROR] Failed to save log: {str(e)}\n")

    def show_statistics(self):
        """Show pipeline statistics in a popup."""
        if self.pipeline:
            stats = self.pipeline.get_statistics()
            stats_text = "\n".join([f"  {k}: {v}" for k, v in stats.items()])
            self._update_log_widget(f"\n[*] Pipeline Statistics:\n{stats_text}\n")
        else:
            self._update_log_widget("[*] No active pipeline. Start capture first.\n")

    def validate_parameters(self):
        """Validate all parameters before starting capture."""
        interface = self.interface_entry.get().strip()
        model_path = self.model_entry.get().strip()
        
        if not interface:
            self._update_log_widget("[ERROR] Network interface is required\n")
            return False
        
        if not model_path:
            self._update_log_widget("[ERROR] Model path is required\n")
            return False
            
        return True

    def start_capture(self):
        """Start the NIDS pipeline."""
        if self.capturing:
            return
            
        if not self.validate_parameters():
            return
            
        interface = self.interface_entry.get().strip()
        model_path = self.model_entry.get().strip()
        
        # Disable UI during startup
        self._set_ui_state(False)
            
        try:
            # Create the pipeline with all 5 components
            self.pipeline = NIDSPipeline(
                interface=interface,
                binary_model_path=model_path,
                multi_class_model_path=PATHS["Multi-class Model"],
                scaler_path=PATHS["Scaler"],
                output_path=PATHS["Output CSV"],
                detection_callback=self._on_detection,
                log_callback=self._update_log_widget
            )
            
            # Start the pipeline
            self.pipeline.start()
            self.capturing = True
            
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            self._update_log_widget(f"[*] Pipeline started on interface {interface}\n")
            self._update_log_widget(f"[*] Binary model: {model_path}\n")
            self._update_log_widget("[*] Components: Capturer → Parser → Extractor → Mapper → Detector\n")
            
            # Start monitoring thread
            threading.Thread(target=self._monitor_capture, daemon=True).start()
            
        except PermissionError:
            self._update_log_widget("[ERROR] Permission denied. Try running with sudo.\n")
            self._set_ui_state(True)
        except Exception as e:
            self._update_log_widget(f"[ERROR] Failed to start capture: {str(e)}\n")
            self._set_ui_state(True)

    def _on_detection(self, result: DetectionResult):
        """Callback for detection results from the Anomaly Detector."""
        # Detection logging is handled in the pipeline
        pass

    def _monitor_capture(self):
        """Monitor packet capture and update statistics."""
        last_packet_count = 0
        last_time = time.time()
        
        while self.capturing and self.pipeline:
            try:
                current_time = time.time()
                elapsed = current_time - last_time
                
                if elapsed >= 1.0:
                    # Get current packet count
                    current_packet_count = self.pipeline.packets_captured
                    
                    # Calculate throughput
                    packets_diff = current_packet_count - last_packet_count
                    self.throughput_value = int(packets_diff / elapsed)
                    self.packet_count = current_packet_count
                    self.flow_count = self.pipeline.active_flows
                    
                    # Update for next iteration
                    last_packet_count = current_packet_count
                    last_time = current_time
                    
                    # Update resource usage
                    process = psutil.Process()
                    self.cpu_usage_value = process.cpu_percent(interval=None)
                    self.memory_usage_value = process.memory_info().rss / 1024 / 1024
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error in monitor: {e}")
                break

    def _set_ui_state(self, enabled):
        """Enable/disable UI elements."""
        state = "normal" if enabled else "disabled"
        self.start_btn.config(state=state)
        self.interface_entry.config(state=state)
        self.model_entry.config(state=state)

    def stop_capture(self):
        """Stop the NIDS pipeline."""
        if self.pipeline and self.capturing:
            self.capturing = False
            
            try:
                # Stop the pipeline (this flushes all flows)
                self.pipeline.stop()
                
                # Log final statistics
                stats = self.pipeline.get_statistics()
                self._update_log_widget(f"[*] Final stats: {stats['packets_captured']} packets, ")
                self._update_log_widget(f"{stats['flows_completed']} flows, ")
                self._update_log_widget(f"{stats['attacks_detected']} attacks detected\n")
                
            except Exception as e:
                self._update_log_widget(f"[ERROR] Error stopping capture: {str(e)}\n")
            
            self.pipeline = None
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self._set_ui_state(True)
            
            self._update_log_widget("[!] NIDS pipeline stopped\n")

    def _update_log_widget(self, message):
        """Update the log widget with a new message."""
        self.log_widget.config(state=tk.NORMAL)
        self.log_widget.insert(tk.END, message)
        self.log_widget.see(tk.END)
        self.log_widget.config(state=tk.DISABLED)

    def clear_log(self):
        """Clear the log widget."""
        self.log_queue.clear()
        self.log_widget.config(state=tk.NORMAL)
        self.log_widget.delete(1.0, tk.END)
        self.log_widget.config(state=tk.DISABLED)

    def update_system_metrics(self):
        """Update system resource usage displays."""
        try:
            battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
            power_status = "Plugged" if battery and battery.power_plugged else "Battery" if battery else "Unknown"
            
            self.cpu_label.config(text=f"CPU: {self.cpu_usage_value:.1f}%")
            self.mem_label.config(text=f"Memory: {self.memory_usage_value:.1f} MB")
            self.power_label.config(text=f"Power: {power_status}")
            self.throughput_label.config(text=f"Throughput: {self.throughput_value} pkt/s")
            self.packet_count_label.config(text=f"Packets: {self.packet_count}")
            self.flow_count_label.config(text=f"Flows: {self.flow_count}")
            
        except Exception as e:
            print(f"Error updating system metrics: {e}")

    def exit_app(self):
        """Clean exit from application."""
        if self.capturing and self.pipeline:
            self.stop_capture()
        
        if self.root:
            self.root.quit()
            self.root.destroy()

    def run(self):
        """Start the GUI application."""
        def periodic_updates():
            while True:
                try:
                    if not self.root.winfo_exists():
                        break
                    
                    self.root.after(0, self.update_system_metrics)
                    time.sleep(self.resource_update_interval)
                    
                except Exception as e:
                    print(f"Error in periodic updates: {e}")
                    break
        
        threading.Thread(target=periodic_updates, daemon=True).start()
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.exit_app()


if __name__ == "__main__":
    app = NIDSGUI()
    app.run()