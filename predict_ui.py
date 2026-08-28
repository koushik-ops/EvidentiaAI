"""
DriveOps Forensic AI — Minimalist Accident & Telemetry Analyzer
Upload any vehicle telemetry file (.xlsx, .csv, .json) to automatically analyze driving behavior,
predict accident causes (Rash driving, Overspeeding, Brake failure, Collision, etc.), and generate a multi-graph PDF report.
"""

import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import test_model
import report_generator


# ── Design Tokens ───────────────────────────────────────────────────────
APP_TITLE = "DriveOps — Vehicle Forensic AI"
BG_MAIN = "#FFFFFF"
BG_CARD = "#F8FBFF"
BLUE_PRIMARY = "#1A73E8"
BLUE_HOVER = "#1557B0"
BLUE_LIGHT = "#E8F0FE"
BLUE_ACCENT = "#D2E3FC"
TEXT_DARK = "#202124"
TEXT_MUTED = "#5F6368"
BORDER_LIGHT = "#DADCE0"
BORDER_FOCUS = "#8AB4F8"

ALERT_RED = "#D93025"
ALERT_AMBER = "#E37400"
ALERT_GREEN = "#1E8E3E"


class ForensicApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1120x840")
        self.root.configure(bg=BG_MAIN)
        self.root.minsize(980, 700)

        # State
        self.selected_file_path = None
        self.analysis_payload = None
        self.auto_pdf_path = None
        self.is_analyzing = False
        self.canvas_widget = None

        self._configure_styles()
        self._build_ui()

    def _configure_styles(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure(".", font=("Segoe UI", 10), background=BG_MAIN, foreground=TEXT_DARK)
        s.configure("TFrame", background=BG_MAIN)
        s.configure("Card.TFrame", background=BG_CARD, relief="flat")
        s.configure("Primary.TButton", background=BLUE_PRIMARY, foreground="white",
                     borderwidth=0, padding=(22, 10), font=("Segoe UI Semibold", 10))
        s.map("Primary.TButton",
              background=[("active", BLUE_HOVER), ("disabled", BLUE_ACCENT)],
              foreground=[("disabled", TEXT_MUTED)])
        
        s.configure("Action.TButton", background=BLUE_PRIMARY, foreground="white",
                     borderwidth=0, padding=(16, 8), font=("Segoe UI Semibold", 9))
        s.map("Action.TButton",
              background=[("active", BLUE_HOVER), ("disabled", BLUE_ACCENT)],
              foreground=[("disabled", TEXT_MUTED)])

        s.configure("Success.TButton", background=ALERT_GREEN, foreground="white",
                     borderwidth=0, padding=(16, 8), font=("Segoe UI Semibold", 9))
        s.map("Success.TButton",
              background=[("active", "#137333"), ("disabled", BLUE_ACCENT)],
              foreground=[("disabled", TEXT_MUTED)])

        s.configure("Secondary.TButton", padding=(14, 7), font=("Segoe UI", 9))
        
        s.configure("Treeview.Heading", font=("Segoe UI Semibold", 9), background=BLUE_LIGHT, foreground=TEXT_DARK)
        s.configure("Treeview", rowheight=24, fieldbackground="white", background="white",
                     foreground=TEXT_DARK, font=("Segoe UI", 9))
        s.map("Treeview", background=[("selected", BLUE_ACCENT)], foreground=[("selected", TEXT_DARK)])

    def _build_ui(self):
        # Top Header
        self._build_header()

        # Main Scrollable Area
        main_container = tk.Frame(self.root, bg=BG_MAIN)
        main_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main_container, bg=BG_MAIN, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas.yview)
        
        self.scroll_frame = tk.Frame(self.canvas, bg=BG_MAIN, padx=36, pady=16)
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Ensure content fills width on resize
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

        # Mouse wheel support
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Section 1: File Upload Box
        self._build_upload_box(self.scroll_frame)

        # Section 2: Forensic Results Container (Hidden until analysis completes)
        self._build_results_dashboard(self.scroll_frame)

        # Status Bar
        self._build_status_bar()

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG_MAIN, padx=36, pady=16)
        header.pack(fill="x")

        tk.Label(header, text="DriveOps Forensic AI", font=("Segoe UI Semibold", 20),
                 bg=BG_MAIN, fg=TEXT_DARK).pack(anchor="w")
        tk.Label(header, text="Automated Vehicle Telemetry Analysis & Accident Cause Forensics",
                 font=("Segoe UI", 10), bg=BG_MAIN, fg=TEXT_MUTED).pack(anchor="w", pady=(2, 0))

        # Thin light-blue divider
        divider = tk.Frame(self.root, bg=BLUE_ACCENT, height=2)
        divider.pack(fill="x")

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=BLUE_LIGHT, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready. Select or upload any telemetry file (.xlsx, .csv, .json) to begin.")
        tk.Label(bar, textvariable=self.status_var, font=("Segoe UI", 9),
                 bg=BLUE_LIGHT, fg=TEXT_MUTED, padx=16).pack(anchor="w", fill="x")

    # ── Upload Box ──────────────────────────────────────────────────────
    def _build_upload_box(self, parent):
        upload_card = tk.Frame(parent, bg=BG_CARD, bd=1, relief="solid",
                               highlightbackground=BORDER_LIGHT, highlightthickness=1, padx=28, pady=22)
        upload_card.pack(fill="x", pady=(0, 16))

        tk.Label(upload_card, text="Upload Vehicle Telemetry Data",
                 font=("Segoe UI Semibold", 13), bg=BG_CARD, fg=TEXT_DARK).pack(anchor="w")
        tk.Label(upload_card,
                 text="Accepts any file format (.xlsx, .csv, .json). Sensor signals are automatically standardized and evaluated.",
                 font=("Segoe UI", 9), bg=BG_CARD, fg=TEXT_MUTED).pack(anchor="w", pady=(2, 14))

        # File Drop/Select Row
        drop_frame = tk.Frame(upload_card, bg="white", bd=1, relief="solid",
                              highlightbackground=BORDER_FOCUS, highlightthickness=1, padx=16, pady=12)
        drop_frame.pack(fill="x", pady=(0, 14))

        self.file_label_var = tk.StringVar(value="No telemetry file selected yet")
        self.file_info_var = tk.StringVar(value="Click 'Browse Telemetry File' to upload")

        info_col = tk.Frame(drop_frame, bg="white")
        info_col.pack(side="left", fill="x", expand=True)

        tk.Label(info_col, textvariable=self.file_label_var, font=("Segoe UI Semibold", 10),
                 bg="white", fg=TEXT_DARK).pack(anchor="w")
        tk.Label(info_col, textvariable=self.file_info_var, font=("Segoe UI", 9),
                 bg="white", fg=TEXT_MUTED).pack(anchor="w")

        browse_btn = ttk.Button(drop_frame, text="Browse Telemetry File", style="Secondary.TButton",
                                command=self._browse_file)
        browse_btn.pack(side="right", padx=(10, 0))

        # Action Button Row
        btn_row = tk.Frame(upload_card, bg=BG_CARD)
        btn_row.pack(fill="x")

        self.analyze_btn = ttk.Button(btn_row, text="Analyze Telemetry & Generate Report",
                                      style="Primary.TButton", command=self._start_analysis, state="disabled")
        self.analyze_btn.pack(side="left")

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Vehicle Telemetry File",
            filetypes=[
                ("All Supported Files", "*.csv;*.xlsx;*.xls;*.json"),
                ("Excel Files", "*.xlsx;*.xls"),
                ("CSV Files", "*.csv"),
                ("JSON Files", "*.json"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return

        self.selected_file_path = path
        filename = os.path.basename(path)
        size_kb = os.path.getsize(path) / 1024.0
        ext = os.path.splitext(path)[1].upper()

        self.file_label_var.set(f"Selected: {filename}")
        self.file_info_var.set(f"Format: {ext}  •  Size: {size_kb:.1f} KB  •  Ready for ML analysis")
        self.analyze_btn.configure(state="normal")
        self.status_var.set(f"File loaded: {filename}. Click 'Analyze Telemetry & Generate Report'.")

    # ── Forensic Results Dashboard ──────────────────────────────────────
    def _build_results_dashboard(self, parent):
        self.results_container = tk.Frame(parent, bg=BG_MAIN)
        self.results_container.pack(fill="x", pady=(0, 16))

        # Initially hidden until first analysis
        self.results_container.pack_forget()

        # 1. Forensic Verdict Banner
        self.verdict_card = tk.Frame(self.results_container, bg=BLUE_PRIMARY, padx=22, pady=16)
        self.verdict_card.pack(fill="x", pady=(0, 14))

        self.cause_title_var = tk.StringVar(value="CAUSE: NORMAL DRIVING")
        self.behavior_badge_var = tk.StringVar(value="BEHAVIOR: NORMAL  |  SEVERITY: LOW")
        self.reason_text_var = tk.StringVar(value="Awaiting analysis...")

        tk.Label(self.verdict_card, textvariable=self.cause_title_var, font=("Segoe UI Semibold", 15),
                 bg=BLUE_PRIMARY, fg="white").pack(anchor="w")
        tk.Label(self.verdict_card, textvariable=self.behavior_badge_var, font=("Segoe UI Semibold", 10),
                 bg=BLUE_PRIMARY, fg=BLUE_LIGHT).pack(anchor="w", pady=(2, 6))
        
        reason_label = tk.Label(self.verdict_card, textvariable=self.reason_text_var, font=("Segoe UI", 10),
                                bg=BLUE_PRIMARY, fg="white", wraplength=940, justify="left")
        reason_label.pack(anchor="w")

        # 2. Key Telemetry Stat Cards
        self.stats_grid = tk.Frame(self.results_container, bg=BG_MAIN)
        self.stats_grid.pack(fill="x", pady=(0, 14))

        self.stat_vars = {
            "max_speed": tk.StringVar(value="—"),
            "peak_g_force": tk.StringVar(value="—"),
            "peak_yaw": tk.StringVar(value="—"),
            "max_brake": tk.StringVar(value="—"),
        }

        cards_spec = [
            ("max_speed", "Max Speed", "km/h"),
            ("peak_g_force", "Peak G-Force", "g"),
            ("peak_yaw", "Peak Yaw / Rotation", "rad/s"),
            ("max_brake", "Peak Brake Engagement", "%"),
        ]

        for i, (key, label, unit) in enumerate(cards_spec):
            c = tk.Frame(self.stats_grid, bg=BLUE_LIGHT, padx=14, pady=10, bd=1,
                         relief="solid", highlightbackground=BLUE_ACCENT, highlightthickness=1)
            c.pack(side="left", fill="x", expand=True, padx=(0 if i == 0 else 8, 0))
            tk.Label(c, textvariable=self.stat_vars[key], font=("Segoe UI Semibold", 14),
                     bg=BLUE_LIGHT, fg=TEXT_DARK).pack(anchor="w")
            tk.Label(c, text=f"{label} ({unit})", font=("Segoe UI", 9),
                     bg=BLUE_LIGHT, fg=TEXT_MUTED).pack(anchor="w")

        # 3. Action Row: Open / Download PDF Report & Export Data
        action_row = tk.Frame(self.results_container, bg=BG_CARD, bd=1, relief="solid",
                              highlightbackground=BORDER_LIGHT, highlightthickness=1, padx=18, pady=12)
        action_row.pack(fill="x", pady=(0, 14))

        tk.Label(action_row, text="Forensic Documentation:", font=("Segoe UI Semibold", 10),
                 bg=BG_CARD, fg=TEXT_DARK).pack(side="left", padx=(0, 12))

        self.open_pdf_btn = ttk.Button(action_row, text="📄 Open PDF Report",
                                       style="Success.TButton", command=self._open_auto_pdf)
        self.open_pdf_btn.pack(side="left", padx=(0, 8))

        self.download_pdf_btn = ttk.Button(action_row, text="Save PDF As...",
                                           style="Action.TButton", command=self._download_pdf_report)
        self.download_pdf_btn.pack(side="left", padx=(0, 8))

        self.export_csv_btn = ttk.Button(action_row, text="Export Raw Data",
                                         style="Secondary.TButton", command=self._export_predictions)
        self.export_csv_btn.pack(side="left")

        self.report_msg_var = tk.StringVar(value="")
        tk.Label(action_row, textvariable=self.report_msg_var, font=("Segoe UI Semibold", 9),
                 bg=BG_CARD, fg=ALERT_GREEN).pack(side="left", padx=(14, 0))

        # 4. Embedded Telemetry Graph Preview
        graph_header = tk.Frame(self.results_container, bg=BG_MAIN)
        graph_header.pack(fill="x", pady=(0, 4))
        tk.Label(graph_header, text="Telemetry Waveform Preview", font=("Segoe UI Semibold", 11),
                 bg=BG_MAIN, fg=BLUE_PRIMARY).pack(anchor="w")

        self.graph_frame = tk.Frame(self.results_container, bg="white", bd=1, relief="solid",
                                    highlightbackground=BORDER_LIGHT, highlightthickness=1, padx=6, pady=6)
        self.graph_frame.pack(fill="x", pady=(0, 14))

        # 5. Prediction Data Table Preview
        table_header = tk.Frame(self.results_container, bg=BG_MAIN)
        table_header.pack(fill="x", pady=(0, 4))
        tk.Label(table_header, text="Window-by-Window Forensic Log", font=("Segoe UI Semibold", 11),
                 bg=BG_MAIN, fg=BLUE_PRIMARY).pack(anchor="w")

        table_card = tk.Frame(self.results_container, bg="white", bd=1, relief="solid",
                              highlightbackground=BORDER_LIGHT, highlightthickness=1)
        table_card.pack(fill="both", expand=True, pady=(0, 10))

        self.preview_tree = ttk.Treeview(table_card, show="headings", height=8)
        self.preview_tree.pack(side="left", fill="both", expand=True)

        y_scroll = ttk.Scrollbar(table_card, orient="vertical", command=self.preview_tree.yview)
        y_scroll.pack(side="right", fill="y")
        self.preview_tree.configure(yscrollcommand=y_scroll.set)

    # ── Analysis Execution ──────────────────────────────────────────────
    def _start_analysis(self):
        if self.is_analyzing or not self.selected_file_path:
            return

        self.is_analyzing = True
        self.analyze_btn.configure(state="disabled")
        self.status_var.set("Analyzing vehicle telemetry & generating multi-graph report…")

        threading.Thread(
            target=self._analysis_worker,
            args=(self.selected_file_path,),
            daemon=True,
        ).start()

    def _analysis_worker(self, input_path):
        try:
            # 1. Run automated analysis pipeline
            payload = test_model.auto_analyze_telemetry(input_path)

            # 2. Auto-generate the comprehensive PDF report
            input_dir = os.path.dirname(input_path) or os.getcwd()
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            auto_pdf_path = os.path.join(input_dir, f"{base_name}_Forensic_Report.pdf")

            report_generator.generate_report(
                results_df=payload["results"],
                summary=payload["summary"],
                output_path=auto_pdf_path,
                input_info={"input_path": payload["input_path"]},
                forensic_insights=payload["forensic_insights"],
                raw_telemetry_df=payload.get("processed_raw_df"),
            )
            payload["auto_pdf_path"] = auto_pdf_path

            self.root.after(0, lambda: self._on_analysis_success(payload))
        except Exception as exc:
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.root.after(0, lambda: self._on_analysis_error(error_text))

    def _on_analysis_success(self, payload):
        self.is_analyzing = False
        self.analyze_btn.configure(state="normal")
        self.analysis_payload = payload
        self.auto_pdf_path = payload.get("auto_pdf_path")

        forensics = payload["forensic_insights"]
        metrics = forensics["telemetry_metrics"]

        # Update Verdict Banner
        cause_display = forensics["primary_cause_display"].upper()
        severity = forensics["severity"]
        behavior = forensics["primary_behavior"]
        reason = forensics["primary_reason"]

        self.cause_title_var.set(f"PREDICTED CAUSE: {cause_display}")
        self.behavior_badge_var.set(f"DRIVING BEHAVIOR: {behavior}  |  INCIDENT SEVERITY: {severity}")
        self.reason_text_var.set(f"Forensic Assessment: {reason}")

        # Update Banner Color according to severity
        bg_banner = ALERT_RED if severity == "CRITICAL" else ALERT_AMBER if severity == "HIGH" else BLUE_PRIMARY
        self.verdict_card.configure(bg=bg_banner)
        for child in self.verdict_card.winfo_children():
            child.configure(bg=bg_banner)

        # Update Stat Cards
        self.stat_vars["max_speed"].set(f"{metrics.get('max_speed', 0):.1f}")
        self.stat_vars["peak_g_force"].set(f"{metrics.get('peak_g_force', 0):.2f}")
        self.stat_vars["peak_yaw"].set(f"{metrics.get('peak_yaw', 0):.3f}")
        self.stat_vars["max_brake"].set(f"{metrics.get('max_brake', 0):.1f}")

        # Render In-App Graph
        self._render_graph_preview(payload.get("processed_raw_df"))

        # Populate Data Table
        self._populate_table(payload["results"])

        # Display dashboard & report notification
        self.results_container.pack(fill="x", pady=(0, 16))
        pdf_fn = os.path.basename(self.auto_pdf_path) if self.auto_pdf_path else "Report.pdf"
        self.report_msg_var.set(f"✓ PDF Report generated: {pdf_fn}")
        self.status_var.set(f"Analysis Complete! Primary Cause: {forensics['primary_cause_display']}. PDF Report Ready.")

    def _on_analysis_error(self, error_text):
        self.is_analyzing = False
        self.analyze_btn.configure(state="normal")
        self.status_var.set("Analysis failed. Please check input file format.")
        messagebox.showerror(APP_TITLE, f"Telemetry Analysis Error:\n\n{error_text}")

    # ── Graph Preview ───────────────────────────────────────────────────
    def _render_graph_preview(self, raw_df):
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
            self.canvas_widget = None

        if raw_df is None or raw_df.empty:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 2.3), dpi=100)
        fig.patch.set_facecolor("white")
        plt.subplots_adjust(wspace=0.25, bottom=0.22, top=0.88, left=0.08, right=0.96)

        samples = np.arange(len(raw_df))

        # Speed Curve
        ax1.set_facecolor("#FAFBFC")
        if "Speed" in raw_df.columns:
            speeds = pd.to_numeric(raw_df["Speed"], errors="coerce").fillna(0.0).values
            ax1.plot(samples, speeds, color=BLUE_PRIMARY, linewidth=1.8, label="Speed")
            ax1.fill_between(samples, speeds, color=BLUE_PRIMARY, alpha=0.1)
            ax1.set_ylabel("Speed (km/h)", fontsize=8, color=TEXT_MUTED, fontweight="bold")
        ax1.set_title("Speed Profile", fontsize=9, fontweight="bold", color=TEXT_DARK)
        ax1.set_xlabel("Sample Index", fontsize=8, color=TEXT_MUTED)
        ax1.grid(True, linestyle="--", alpha=0.4, color=BORDER_LIGHT)
        ax1.tick_params(labelsize=7, colors=TEXT_MUTED)

        # Acceleration G-Force Curve
        ax2.set_facecolor("#FAFBFC")
        acc_col = "LinAccMag" if "LinAccMag" in raw_df.columns else "AccMag" if "AccMag" in raw_df.columns else "AccX"
        if acc_col in raw_df.columns:
            accs = pd.to_numeric(raw_df[acc_col], errors="coerce").fillna(0.0).values
            ax2.plot(samples, accs, color=ALERT_RED, linewidth=1.6, label="Acc G-Force")
            ax2.fill_between(samples, accs, color=ALERT_RED, alpha=0.1)
            ax2.set_ylabel("G-Force (g)", fontsize=8, color=TEXT_MUTED, fontweight="bold")
        ax2.set_title("G-Force & Acceleration Spikes", fontsize=9, fontweight="bold", color=TEXT_DARK)
        ax2.set_xlabel("Sample Index", fontsize=8, color=TEXT_MUTED)
        ax2.grid(True, linestyle="--", alpha=0.4, color=BORDER_LIGHT)
        ax2.tick_params(labelsize=7, colors=TEXT_MUTED)

        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)

    # ── Table Population ────────────────────────────────────────────────
    def _populate_table(self, df):
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)

        if df.empty:
            self.preview_tree.configure(columns=())
            return

        display_cols = [
            ("window_index", "Window", 60),
            ("prediction_label", "Behavior", 90),
            ("predicted_cause", "Predicted Cause", 170),
            ("cause_confidence", "Confidence", 90),
            ("cause_reason", "Forensic Signal Reasoning", 450),
        ]

        valid_cols = [col for col, _, _ in display_cols if col in df.columns]
        self.preview_tree.configure(columns=valid_cols)

        for col, heading, width in display_cols:
            if col in valid_cols:
                self.preview_tree.heading(col, text=heading)
                self.preview_tree.column(col, width=width, anchor="w")

        for _, row in df.head(30).iterrows():
            values = []
            for col, _, _ in display_cols:
                if col in valid_cols:
                    val = row.get(col, "")
                    if col == "predicted_cause":
                        val = str(val).replace("_", " ").title()
                    elif col == "cause_confidence" and pd.notna(val):
                        val = f"{float(val):.2f}"
                    values.append(str(val))
            self.preview_tree.insert("", "end", values=values)

    # ── Report Opening & Saving ─────────────────────────────────────────
    def _open_auto_pdf(self):
        if self.auto_pdf_path and os.path.exists(self.auto_pdf_path):
            try:
                os.startfile(self.auto_pdf_path)
            except Exception as e:
                messagebox.showerror(APP_TITLE, f"Could not open PDF file:\n{e}")
        else:
            messagebox.showinfo(APP_TITLE, "PDF Report is not available yet.")

    def _download_pdf_report(self):
        if not self.analysis_payload:
            messagebox.showinfo(APP_TITLE, "Please analyze a file first.")
            return

        base_name = os.path.splitext(os.path.basename(self.analysis_payload["input_path"]))[0]
        default_fn = f"DriveOps_Forensic_Report_{base_name}.pdf"

        save_path = filedialog.asksaveasfilename(
            title="Save Forensic PDF Report",
            defaultextension=".pdf",
            initialfile=default_fn,
            filetypes=[("PDF Documents", "*.pdf")],
        )
        if not save_path:
            return

        self.download_pdf_btn.configure(state="disabled")
        self.report_msg_var.set("Generating custom PDF report…")

        threading.Thread(
            target=self._pdf_worker,
            args=(save_path,),
            daemon=True,
        ).start()

    def _pdf_worker(self, save_path):
        try:
            report_generator.generate_report(
                results_df=self.analysis_payload["results"],
                summary=self.analysis_payload["summary"],
                output_path=save_path,
                input_info={"input_path": self.analysis_payload["input_path"]},
                forensic_insights=self.analysis_payload["forensic_insights"],
                raw_telemetry_df=self.analysis_payload.get("processed_raw_df"),
            )
            self.root.after(0, lambda: self._on_pdf_ready(save_path))
        except Exception as exc:
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.root.after(0, lambda: self._on_pdf_error(error_text))

    def _on_pdf_ready(self, save_path):
        self.download_pdf_btn.configure(state="normal")
        self.auto_pdf_path = save_path
        self.report_msg_var.set(f"✓ PDF Saved to {os.path.basename(save_path)}")
        self.status_var.set(f"Report saved to {save_path}")
        try:
            os.startfile(save_path)
        except Exception:
            pass

    def _on_pdf_error(self, error_text):
        self.download_pdf_btn.configure(state="normal")
        self.report_msg_var.set("PDF generation failed.")
        messagebox.showerror(APP_TITLE, f"Failed to generate PDF Report:\n\n{error_text}")

    def _export_predictions(self):
        if not self.analysis_payload:
            return

        base_name = os.path.splitext(os.path.basename(self.analysis_payload["input_path"]))[0]
        save_path = filedialog.asksaveasfilename(
            title="Export Prediction Results",
            defaultextension=".csv",
            initialfile=f"{base_name}_predictions.csv",
            filetypes=[("CSV File", "*.csv"), ("Excel File", "*.xlsx"), ("JSON File", "*.json")],
        )
        if not save_path:
            return

        try:
            test_model.save_output_file(self.analysis_payload["results"], save_path)
            messagebox.showinfo(APP_TITLE, f"Predictions successfully exported to:\n{save_path}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Failed to export predictions:\n{exc}")


def main():
    root = tk.Tk()
    ForensicApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
