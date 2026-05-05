import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

import test_model


APP_TITLE = "DriveOps Predictor"
APP_BG = "#f3f6f9"
PANEL_BG = "#ffffff"
ACCENT = "#0f766e"
TEXT = "#102a43"
MUTED = "#5c6f82"


class PredictorUI:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x820")
        self.root.configure(bg=APP_BG)
        self.root.minsize(1080, 720)

        self.input_mode_var = tk.StringVar(value="raw")
        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.config_path_var = tk.StringVar(value=test_model.DEFAULT_HEURISTIC_CONFIG_PATH)
        self.cause_label_col_var = tk.StringVar()
        self.window_size_var = tk.StringVar(value=str(test_model.DEFAULT_WINDOW_SIZE))
        self.step_size_var = tk.StringVar(value=str(test_model.DEFAULT_STEP_SIZE))
        self.preview_rows_var = tk.StringVar(value="20")
        self.status_var = tk.StringVar(value="Choose a CSV file, then click Predict.")
        self.summary_var = tk.StringVar(value="No prediction has been run yet.")

        self.current_output_path = None
        self.current_preview_df = pd.DataFrame()
        self.is_running = False

        self._configure_styles()
        self._build_layout()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        default_font = ("Segoe UI", 10)
        heading_font = ("Segoe UI Semibold", 11)
        title_font = ("Segoe UI Semibold", 16)

        style.configure(".", font=default_font)
        style.configure("App.TFrame", background=APP_BG)
        style.configure("Panel.TFrame", background=PANEL_BG, relief="flat")
        style.configure("Title.TLabel", background=APP_BG, foreground=TEXT, font=title_font)
        style.configure("Hint.TLabel", background=APP_BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=PANEL_BG, foreground=TEXT, font=heading_font)
        style.configure("Body.TLabel", background=PANEL_BG, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL_BG, foreground=MUTED)
        style.configure("Accent.TButton", background=ACCENT, foreground="white", borderwidth=0, padding=(14, 8))
        style.map(
            "Accent.TButton",
            background=[("active", "#115e59"), ("disabled", "#8fb9b5")],
            foreground=[("disabled", "#eef6f5")],
        )
        style.configure("Secondary.TButton", padding=(12, 7))
        style.configure("Mode.TRadiobutton", background=PANEL_BG, foreground=TEXT)
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10))
        style.configure("Treeview", rowheight=26, fieldbackground="white")

    def _build_layout(self):
        outer = ttk.Frame(self.root, style="App.TFrame", padding=20)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x", pady=(0, 14))
        ttk.Label(header, text="DriveOps Accident Cause Predictor", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Feed a raw sensor CSV or feature CSV and preview the model output without using the terminal.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        control_panel = ttk.Frame(outer, style="Panel.TFrame", padding=18)
        control_panel.pack(fill="x")

        preview_panel = ttk.Frame(outer, style="Panel.TFrame", padding=18)
        preview_panel.pack(fill="both", expand=True, pady=(16, 0))

        self._build_controls(control_panel)
        self._build_preview(preview_panel)

    def _build_controls(self, parent):
        ttk.Label(parent, text="Prediction Setup", style="PanelTitle.TLabel").grid(
            row=0, column=0, columnspan=6, sticky="w", pady=(0, 12)
        )

        self._add_labeled_entry(
            parent,
            row=1,
            label="Input CSV",
            textvariable=self.input_path_var,
            browse_command=self._browse_input_csv,
            browse_label="Browse",
        )
        self._add_labeled_entry(
            parent,
            row=2,
            label="Output CSV",
            textvariable=self.output_path_var,
            browse_command=self._browse_output_csv,
            browse_label="Save As",
        )
        self._add_labeled_entry(
            parent,
            row=3,
            label="Heuristic Config",
            textvariable=self.config_path_var,
            browse_command=self._browse_config,
            browse_label="Browse",
        )

        ttk.Label(parent, text="Input Type", style="Body.TLabel").grid(row=4, column=0, sticky="w", pady=(10, 4))
        mode_row = ttk.Frame(parent, style="Panel.TFrame")
        mode_row.grid(row=4, column=1, columnspan=5, sticky="w", pady=(10, 4))
        ttk.Radiobutton(
            mode_row,
            text="Raw Sensor CSV",
            variable=self.input_mode_var,
            value="raw",
            style="Mode.TRadiobutton",
            command=self._sync_default_output_path,
        ).pack(side="left", padx=(0, 18))
        ttk.Radiobutton(
            mode_row,
            text="Feature CSV",
            variable=self.input_mode_var,
            value="feature",
            style="Mode.TRadiobutton",
            command=self._sync_default_output_path,
        ).pack(side="left")

        ttk.Label(parent, text="Cause Label Column", style="Body.TLabel").grid(row=5, column=0, sticky="w", pady=(10, 4))
        ttk.Entry(parent, textvariable=self.cause_label_col_var).grid(row=5, column=1, sticky="ew", pady=(10, 4), padx=(0, 12))

        ttk.Label(parent, text="Window Size", style="Body.TLabel").grid(row=5, column=2, sticky="w", pady=(10, 4))
        ttk.Entry(parent, textvariable=self.window_size_var, width=10).grid(row=5, column=3, sticky="w", pady=(10, 4), padx=(0, 12))

        ttk.Label(parent, text="Step Size", style="Body.TLabel").grid(row=5, column=4, sticky="w", pady=(10, 4))
        ttk.Entry(parent, textvariable=self.step_size_var, width=10).grid(row=5, column=5, sticky="w", pady=(10, 4))

        ttk.Label(parent, text="Preview Rows", style="Body.TLabel").grid(row=6, column=0, sticky="w", pady=(10, 4))
        ttk.Entry(parent, textvariable=self.preview_rows_var, width=10).grid(row=6, column=1, sticky="w", pady=(10, 4))

        button_row = ttk.Frame(parent, style="Panel.TFrame")
        button_row.grid(row=7, column=0, columnspan=6, sticky="ew", pady=(16, 2))
        self.run_button = ttk.Button(button_row, text="Predict", style="Accent.TButton", command=self._start_prediction)
        self.run_button.pack(side="left")
        self.open_button = ttk.Button(
            button_row,
            text="Open Saved CSV",
            style="Secondary.TButton",
            command=self._open_saved_csv,
            state="disabled",
        )
        self.open_button.pack(side="left", padx=(10, 0))

        ttk.Label(parent, textvariable=self.status_var, style="Muted.TLabel", wraplength=980).grid(
            row=8, column=0, columnspan=6, sticky="w", pady=(14, 2)
        )

        for column in range(6):
            parent.columnconfigure(column, weight=1 if column in {1, 3, 5} else 0)

    def _build_preview(self, parent):
        ttk.Label(parent, text="Prediction Preview", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(parent, textvariable=self.summary_var, style="Muted.TLabel", wraplength=1100).pack(anchor="w", pady=(6, 14))

        columns_frame = ttk.Frame(parent, style="Panel.TFrame")
        columns_frame.pack(fill="both", expand=True)

        self.preview_tree = ttk.Treeview(columns_frame, show="headings")
        self.preview_tree.pack(side="left", fill="both", expand=True)

        y_scroll = ttk.Scrollbar(columns_frame, orient="vertical", command=self.preview_tree.yview)
        y_scroll.pack(side="right", fill="y")
        x_scroll = ttk.Scrollbar(parent, orient="horizontal", command=self.preview_tree.xview)
        x_scroll.pack(fill="x", pady=(8, 0))

        self.preview_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    def _add_labeled_entry(self, parent, row, label, textvariable, browse_command, browse_label):
        ttk.Label(parent, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=textvariable)
        entry.grid(row=row, column=1, columnspan=4, sticky="ew", pady=4, padx=(0, 12))
        ttk.Button(parent, text=browse_label, style="Secondary.TButton", command=browse_command).grid(
            row=row, column=5, sticky="e", pady=4
        )

    def _browse_input_csv(self):
        path = filedialog.askopenfilename(
            title="Choose Input CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.input_path_var.set(path)
        self._sync_default_output_path(force=True)
        self.status_var.set("Input file selected. Ready to predict.")

    def _browse_output_csv(self):
        default_name = os.path.basename(self.output_path_var.get() or "predictions.csv")
        path = filedialog.asksaveasfilename(
            title="Choose Output CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.output_path_var.set(path)

    def _browse_config(self):
        path = filedialog.askopenfilename(
            title="Choose Heuristic Config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.config_path_var.set(path)

    def _sync_default_output_path(self, force=False):
        input_path = self.input_path_var.get().strip()
        if not input_path:
            return
        suggested = test_model.default_output_path(input_path)
        current = self.output_path_var.get().strip()
        if force or not current:
            self.output_path_var.set(suggested)

    def _start_prediction(self):
        if self.is_running:
            return

        input_path = self.input_path_var.get().strip()
        output_path = self.output_path_var.get().strip()
        config_path = self.config_path_var.get().strip() or test_model.DEFAULT_HEURISTIC_CONFIG_PATH

        if not input_path:
            messagebox.showerror(APP_TITLE, "Choose an input CSV first.")
            return
        if not os.path.exists(input_path):
            messagebox.showerror(APP_TITLE, "The selected input CSV does not exist.")
            return
        if not output_path:
            self._sync_default_output_path(force=True)
            output_path = self.output_path_var.get().strip()
        if not config_path or not os.path.exists(config_path):
            messagebox.showerror(APP_TITLE, "The heuristic config file was not found.")
            return

        try:
            window_size = int(self.window_size_var.get().strip())
            step_size = int(self.step_size_var.get().strip())
            preview_rows = int(self.preview_rows_var.get().strip())
        except ValueError:
            messagebox.showerror(APP_TITLE, "Window size, step size, and preview rows must be whole numbers.")
            return

        if window_size <= 0 or step_size <= 0 or preview_rows <= 0:
            messagebox.showerror(APP_TITLE, "Window size, step size, and preview rows must be greater than zero.")
            return

        self.is_running = True
        self.run_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.status_var.set("Running prediction... large files may take a few seconds.")
        self.summary_var.set("Working on your file...")

        worker = threading.Thread(
            target=self._run_prediction_worker,
            args=(
                input_path,
                output_path,
                config_path,
                self.input_mode_var.get(),
                self.cause_label_col_var.get().strip() or None,
                window_size,
                step_size,
                preview_rows,
            ),
            daemon=True,
        )
        worker.start()

    def _run_prediction_worker(
        self,
        input_path,
        output_path,
        config_path,
        input_mode,
        cause_label_col,
        window_size,
        step_size,
        preview_rows,
    ):
        try:
            prediction = test_model.predict_from_input_path(
                input_path=input_path,
                input_mode=input_mode,
                config_path=config_path,
                window_size=window_size,
                step_size=step_size,
                cause_label_col=cause_label_col,
                output_path=output_path,
            )
            payload = {
                "input_path": prediction["input_path"],
                "output_path": prediction["output_path"],
                "input_kind": prediction["input_kind"],
                "config_path": prediction["config_path"],
                "summary": prediction["summary"],
                "preview_df": prediction["results"].head(preview_rows).copy(),
            }
            self.root.after(0, lambda: self._finish_prediction(payload))
        except Exception as exc:  # pragma: no cover - UI flow
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.root.after(0, lambda: self._fail_prediction(error_text))

    def _finish_prediction(self, payload):
        self.is_running = False
        self.run_button.configure(state="normal")
        self.open_button.configure(state="normal")
        self.current_output_path = payload["output_path"]
        self.current_preview_df = payload["preview_df"]

        self.status_var.set(
            f"Prediction complete. {payload['summary']['rows']:,} row(s) were written to {payload['output_path']}."
        )
        self.summary_var.set(self._format_summary(payload))
        self._populate_preview(payload["preview_df"])

    def _fail_prediction(self, error_text):
        self.is_running = False
        self.run_button.configure(state="normal")
        self.open_button.configure(state="disabled" if not self.current_output_path else "normal")
        self.status_var.set("Prediction failed. See the error dialog for details.")
        self.summary_var.set("No new results were produced.")
        messagebox.showerror(APP_TITLE, error_text)

    def _format_summary(self, payload):
        summary = payload["summary"]
        lines = [
            f"Input: {payload['input_kind']} | Rows generated: {summary['rows']:,}",
            f"Saved CSV: {payload['output_path']}",
        ]

        behavior_counts = summary.get("behavior_counts", {})
        if behavior_counts:
            behavior_text = ", ".join(f"{label}: {count}" for label, count in behavior_counts.items())
            lines.append(f"Behavior predictions: {behavior_text}")

        cause_counts = summary.get("cause_counts", {})
        if cause_counts:
            cause_text = ", ".join(f"{label}: {count}" for label, count in cause_counts.items())
            lines.append(f"Cause predictions: {cause_text}")

        behavior_conf = summary.get("behavior_confidence_mean")
        if behavior_conf is not None and not pd.isna(behavior_conf):
            lines.append(f"Mean behavior confidence: {behavior_conf:.3f}")

        cause_conf = summary.get("cause_confidence_mean")
        if cause_conf is not None and not pd.isna(cause_conf):
            lines.append(f"Mean cause confidence: {cause_conf:.3f}")

        if payload["summary"]["input_mode"] == "raw":
            lines.append(f"Heuristic config: {payload['config_path']}")

        return "\n".join(lines)

    def _populate_preview(self, df):
        for item_id in self.preview_tree.get_children():
            self.preview_tree.delete(item_id)

        if df.empty:
            self.preview_tree.configure(columns=())
            return

        columns = list(df.columns)
        self.preview_tree.configure(columns=columns)

        for column in columns:
            self.preview_tree.heading(column, text=column)
            width = min(max(110, len(column) * 9), 220)
            self.preview_tree.column(column, width=width, anchor="w", stretch=True)

        for row in df.itertuples(index=False):
            values = [self._format_cell(value) for value in row]
            self.preview_tree.insert("", "end", values=values)

    def _format_cell(self, value):
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _open_saved_csv(self):
        if not self.current_output_path or not os.path.exists(self.current_output_path):
            messagebox.showinfo(APP_TITLE, "No saved CSV is available yet.")
            return
        os.startfile(self.current_output_path)  # type: ignore[attr-defined]


def main():
    root = tk.Tk()
    PredictorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
