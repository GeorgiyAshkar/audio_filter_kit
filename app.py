"""
frontend/app.py
================

This module provides the PyQt5 graphical user interface for the DAS
speech filter explorer.  It separates presentation logic from the
processing backend, which lives in the :mod:`backend` package.  All
signal processing, filter definitions and metric calculations reside
in the backend, making it easy to extend or replace the core with
alternative microservices while keeping the GUI layer thin.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import itertools
import tempfile
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import librosa  # type: ignore
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from scipy.signal import welch

# Adjust the Python path so the sibling `backend` package can be imported when
# running this script from the ``frontend`` directory.  Without this, the
# ``backend`` package may not be found on the import path when executing
# ``python app.py`` directly.  We insert the parent directory of this file
# at the front of ``sys.path`` so that ``import backend`` resolves to
# the sibling package shipped with this application.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.dirname(__file__)))

from backend.core import (
    ensure_mono,
    resample_if_needed,
    normalize_pairing_key,
    get_filter_classes,
    run_pipeline,
    compute_metrics,
    total_latency_ms,
    composite_score,
    FilterBase,
)

###############################################################################
# Matplotlib canvas wrapper
###############################################################################


class MplCanvas(FigureCanvas):
    """
    Matplotlib canvas that holds three axes: a small overview (panner),
    the main waveform plot and the power spectrum plot.  The panner is
    used to select a region of the audio.  The waveform displays the
    raw, reference and processed signals, and highlights the selected
    region.  The spectrum shows the PSD of the selected (or full)
    segment.
    """

    def __init__(self):
        # Create figure with three vertical subplots.  The panner takes
        # up a small portion of the height, while the waveform and
        # spectrum share the remaining space.
        self.fig = Figure(figsize=(10, 8))
        super().__init__(self.fig)
        # Use a GridSpec to set relative heights: 1 for panner,
        # 4 for waveform and 4 for spectrum.
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 4, 4])
        self.ax_pan = self.fig.add_subplot(gs[0, 0])
        self.ax_wave = self.fig.add_subplot(gs[1, 0])
        self.ax_spec = self.fig.add_subplot(gs[2, 0])
        self.fig.tight_layout()


###############################################################################
# Main application window
###############################################################################


class AudioFilterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAS Speech Filter Explorer (Modular)")
        self.resize(1650, 950)

        # state variables
        self.raw_signal: np.ndarray = np.array([], dtype=np.float32)
        self.raw_sr: int = 0
        self.ref_signal: np.ndarray = np.array([], dtype=np.float32)
        self.ref_sr: int = 0
        self.processed_signal: np.ndarray = np.array([], dtype=np.float32)
        self.raw_path: str = ""
        self.ref_path: str = ""
        self.raw_folder: str = ""
        self.ref_folder: str = ""
        self.transcripts_folder: str = ""
        self.pipeline: List[FilterBase] = []
        self.search_results: List[Dict[str, Any]] = []
        self.temp_files: List[str] = []
        # load available filters from backend
        self.available_filters: List[type] = get_filter_classes()
        # media player
        self.player = QMediaPlayer()
        self.player.positionChanged.connect(self.on_position_changed)
        # playback state
        self.playback_x: float = 0.0
        self.playback_line = None
        # selection for panning (start_time, end_time) in seconds
        self.selection: Optional[Tuple[float, float]] = None
        # build UI
        self._build_ui()

    # ------------------- UI layout -------------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # file buttons and pipeline save/load
        file_bar = QHBoxLayout()
        self.btn_load_raw = QPushButton("Загрузить noisy/raw")
        self.btn_load_ref = QPushButton("Загрузить reference")
        self.btn_load_raw_folder = QPushButton("Папка raw")
        self.btn_load_ref_folder = QPushButton("Папка ref")
        self.btn_save_pipeline = QPushButton("Сохранить pipeline")
        self.btn_load_pipeline = QPushButton("Открыть pipeline")
        self.btn_load_raw.clicked.connect(self.load_raw_file)
        self.btn_load_ref.clicked.connect(self.load_ref_file)
        self.btn_load_raw_folder.clicked.connect(self.select_raw_folder)
        self.btn_load_ref_folder.clicked.connect(self.select_ref_folder)
        self.btn_save_pipeline.clicked.connect(self.save_pipeline)
        self.btn_load_pipeline.clicked.connect(self.load_pipeline)
        for b in [
            self.btn_load_raw,
            self.btn_load_ref,
            self.btn_load_raw_folder,
            self.btn_load_ref_folder,
            self.btn_save_pipeline,
            self.btn_load_pipeline,
        ]:
            file_bar.addWidget(b)
        layout.addLayout(file_bar)

        # info labels
        self.lbl_raw = QLabel("Raw: —")
        self.lbl_ref = QLabel("Reference: —")
        self.lbl_batch = QLabel("Dataset folders: —")
        self.lbl_transcripts = QLabel("Transcripts folder: —")
        layout.addWidget(self.lbl_raw)
        layout.addWidget(self.lbl_ref)
        layout.addWidget(self.lbl_batch)
        layout.addWidget(self.lbl_transcripts)

        # transcripts folder selection
        transcripts_bar = QHBoxLayout()
        self.btn_load_transcripts = QPushButton("Папка transcripts")
        self.btn_load_transcripts.clicked.connect(self.select_transcripts_folder)
        transcripts_bar.addWidget(self.btn_load_transcripts)
        layout.addLayout(transcripts_bar)

        # splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # left panel: filters and pipeline
        left = QWidget()
        l_layout = QVBoxLayout(left)
        l_layout.addWidget(QLabel("Доступные фильтры (выберите для поиска):"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for cls in self.available_filters:
            item = QListWidgetItem(cls.name)
            item.setData(Qt.UserRole, cls)
            self.available_list.addItem(item)
        l_layout.addWidget(self.available_list)
        # pipeline controls
        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Добавить →")
        self.btn_remove = QPushButton("Удалить")
        self.btn_up = QPushButton("Вверх")
        self.btn_down = QPushButton("Вниз")
        self.btn_add.clicked.connect(self.add_selected_filter)
        self.btn_remove.clicked.connect(self.remove_pipeline_filter)
        self.btn_up.clicked.connect(self.move_up)
        self.btn_down.clicked.connect(self.move_down)
        for b in [self.btn_add, self.btn_remove, self.btn_up, self.btn_down]:
            btn_row.addWidget(b)
        l_layout.addLayout(btn_row)
        # pipeline list
        l_layout.addWidget(QLabel("Pipeline:"))
        self.pipeline_list = QListWidget()
        self.pipeline_list.currentRowChanged.connect(self.refresh_param_editor)
        l_layout.addWidget(self.pipeline_list)
        splitter.addWidget(left)

        # middle panel: parameter editing and actions
        middle = QWidget()
        m_layout = QVBoxLayout(middle)
        # parameter group
        params_group = QGroupBox("Параметры выбранного фильтра")
        self.params_form = QFormLayout(params_group)
        m_layout.addWidget(params_group)
        # action buttons
        actions = QGroupBox("Применение и оптимизация")
        actions_layout = QVBoxLayout(actions)
        self.btn_apply = QPushButton("Применить pipeline")
        self.btn_optimize_current = QPushButton("Перебор параметров текущего pipeline")
        self.btn_eval_batch = QPushButton("Оценить pipeline на папках raw/ref")
        self.btn_apply.clicked.connect(self.apply_pipeline_clicked)
        self.btn_optimize_current.clicked.connect(self.optimize_current_pipeline)
        self.btn_eval_batch.clicked.connect(self.evaluate_dataset_folders)
        actions_layout.addWidget(self.btn_apply)
        actions_layout.addWidget(self.btn_optimize_current)
        actions_layout.addWidget(self.btn_eval_batch)
        m_layout.addWidget(actions)
        # ASR/WER evaluation
        asr_group = QGroupBox("ASR/WER оценка")
        asr_layout = QVBoxLayout(asr_group)
        self.btn_eval_wer_pair = QPushButton("Оценить WER для текущего файла")
        self.btn_eval_wer_dataset = QPushButton("Оценить WER для папок")
        self.btn_eval_wer_pair.clicked.connect(self.evaluate_wer_pair)
        self.btn_eval_wer_dataset.clicked.connect(self.evaluate_wer_dataset)
        asr_layout.addWidget(self.btn_eval_wer_pair)
        asr_layout.addWidget(self.btn_eval_wer_dataset)
        m_layout.addWidget(asr_group)
        # playback group
        playback = QGroupBox("Прослушивание")
        playback_layout = QHBoxLayout(playback)
        self.btn_play_raw = QPushButton("Raw")
        self.btn_play_processed = QPushButton("Processed")
        self.btn_play_ref = QPushButton("Reference")
        self.btn_pause = QPushButton("Пауза")
        self.btn_play_raw.clicked.connect(self.play_raw)
        self.btn_play_processed.clicked.connect(self.play_processed)
        self.btn_play_ref.clicked.connect(self.play_reference)
        self.btn_pause.clicked.connect(self.pause_playback)
        for b in [self.btn_play_raw, self.btn_play_processed, self.btn_play_ref, self.btn_pause]:
            playback_layout.addWidget(b)
        m_layout.addWidget(playback)
        # search group
        search_group = QGroupBox("Brute-force поиск комбинаций")
        search_form = QFormLayout(search_group)
        self.spin_depth = QSpinBox(); self.spin_depth.setRange(1, 4); self.spin_depth.setValue(2)
        self.spin_topk = QSpinBox(); self.spin_topk.setRange(1, 20); self.spin_topk.setValue(8)
        self.spin_maxeval = QSpinBox(); self.spin_maxeval.setRange(10, 20000); self.spin_maxeval.setValue(3000)
        self.btn_search = QPushButton("Поиск лучших цепочек")
        self.btn_apply_result = QPushButton("Применить выбранный результат")
        self.btn_export_results = QPushButton("Экспорт результатов")
        self.btn_search.clicked.connect(self.search_best_pipelines)
        self.btn_apply_result.clicked.connect(self.apply_selected_search_result)
        self.btn_export_results.clicked.connect(self.export_results)
        search_form.addRow("Макс. длина цепочки", self.spin_depth)
        search_form.addRow("Top-K", self.spin_topk)
        search_form.addRow("Лимит оценок", self.spin_maxeval)
        search_form.addRow(self.btn_search)
        search_form.addRow(self.btn_apply_result)
        search_form.addRow(self.btn_export_results)
        m_layout.addWidget(search_group)
        # results list
        m_layout.addWidget(QLabel("Результаты поиска:"))
        self.results_list = QListWidget()
        m_layout.addWidget(self.results_list)
        # metrics labels
        self.lbl_metrics = QLabel("Metrics: —")
        self.lbl_latency = QLabel("Latency: —")
        m_layout.addWidget(self.lbl_metrics)
        m_layout.addWidget(self.lbl_latency)
        # status label
        self.lbl_status = QLabel("Status: ready")
        m_layout.addWidget(self.lbl_status)
        # help button
        self.btn_help = QPushButton("Справка")
        self.btn_help.clicked.connect(self.show_help)
        m_layout.addWidget(self.btn_help)
        splitter.addWidget(middle)

        # right panel: plot
        right = QWidget()
        r_layout = QVBoxLayout(right)
        self.canvas = MplCanvas()
        r_layout.addWidget(self.canvas)
        # Enable interactive selection on both the panner (overview) and
        # waveform axes.  Selecting on the panner provides a high‑level
        # overview for choosing a region, while selecting on the
        # waveform allows fine adjustment.  Both selectors update
        # ``self.selection`` and redraw the plots.
        try:
            # Selector on the main waveform for fine adjustment
            self.span_selector_wave = SpanSelector(
                self.canvas.ax_wave,
                self.on_select_span,
                'horizontal',
                useblit=True,
            )
        except Exception:
            self.span_selector_wave = None
        try:
            # Selector on the panner for coarse selection
            self.span_selector_pan = SpanSelector(
                self.canvas.ax_pan,
                self.on_select_pan,
                'horizontal',
                useblit=True,
            )
        except Exception:
            self.span_selector_pan = None
        splitter.addWidget(right)
        splitter.setSizes([360, 430, 760])

        # initialise transcripts label
        self.update_transcripts_label()

    # ------------------- File operations -------------------
    def load_audio(self, title: str) -> Tuple[np.ndarray, int, str]:
        path, _ = QFileDialog.getOpenFileName(self, title, "", "Audio (*.wav *.mp3 *.m4a *.flac *.ogg *.aac)")
        if not path:
            return np.array([], dtype=np.float32), 0, ""
        y, sr = librosa.load(path, sr=None, mono=True)
        return ensure_mono(y), int(sr), path

    def load_raw_file(self):
        y, sr, path = self.load_audio("Выберите noisy/raw файл")
        if sr > 0:
            self.raw_signal, self.raw_sr, self.raw_path = y, sr, path
            self.lbl_raw.setText(f"Raw: {os.path.basename(path)} | sr={sr} | samples={len(y)}")
            self.redraw()

    def load_ref_file(self):
        y, sr, path = self.load_audio("Выберите reference файл")
        if sr > 0:
            self.ref_signal, self.ref_sr, self.ref_path = y, sr, path
            self.lbl_ref.setText(f"Reference: {os.path.basename(path)} | sr={sr} | samples={len(y)}")
            self.redraw()

    def select_raw_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Папка с raw/noisy файлами")
        if path:
            self.raw_folder = path
            self.update_batch_label()

    def select_ref_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Папка с reference файлами")
        if path:
            self.ref_folder = path
            self.update_batch_label()

    def update_batch_label(self):
        raw = os.path.basename(self.raw_folder) if self.raw_folder else "—"
        ref = os.path.basename(self.ref_folder) if self.ref_folder else "—"
        self.lbl_batch.setText(f"Dataset folders: raw={raw}, ref={ref}")

    def save_pipeline(self):
        if not self.pipeline:
            QMessageBox.information(self, "Пусто", "Pipeline пуст.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить pipeline", "pipeline.json", "JSON (*.json)")
        if not path:
            return
        payload = [f.to_dict() for f in self.pipeline]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть pipeline", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.pipeline = [FilterBase.from_dict(x) for x in payload]
            self.refresh_pipeline_list()
            self.refresh_param_editor(self.pipeline_list.currentRow())
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть pipeline:\n{e}")

    # ------------------- Transcripts operations -------------------
    def select_transcripts_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Папка с транскриптами (.txt)")
        if path:
            self.transcripts_folder = path
            self.update_transcripts_label()

    def update_transcripts_label(self):
        name = os.path.basename(self.transcripts_folder) if self.transcripts_folder else "—"
        self.lbl_transcripts.setText(f"Transcripts folder: {name}")

    def find_transcript_file(self, audio_path: str) -> Optional[str]:
        if not self.transcripts_folder:
            return None
        base_key = normalize_pairing_key(audio_path)
        for fname in os.listdir(self.transcripts_folder):
            if not fname.lower().endswith('.txt'):
                continue
            key = normalize_pairing_key(fname)
            if key == base_key:
                return os.path.join(self.transcripts_folder, fname)
        return None

    def transcribe_audio(self, y: np.ndarray, sr: int) -> str:
        try:
            import speech_recognition as sr_rec
        except ImportError:
            raise RuntimeError("Модуль speech_recognition не установлен. Установите его для использования ASR.")
        recognizer = sr_rec.Recognizer()
        if len(y) == 0:
            return ""
        norm = np.clip(y / max(np.max(np.abs(y)), 1e-8), -1.0, 1.0)
        pcm = (norm * 32767.0).astype(np.int16).tobytes()
        audio_data = sr_rec.AudioData(pcm, sr, 2)
        try:
            text = recognizer.recognize_sphinx(audio_data, language='ru-RU')
        except sr_rec.UnknownValueError:
            return ""
        except sr_rec.RequestError as e:
            raise RuntimeError(f"Ошибка при вызове recognizer: {e}")
        return text.lower()

    def compute_wer(self, truth: str, hyp: str) -> float:
        try:
            import jiwer
        except ImportError:
            raise RuntimeError("Модуль jiwer не установлен. Установите его для вычисления WER.")
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])
        return float(jiwer.wer(truth, hyp, truth_transform=transformation, hypothesis_transform=transformation))

    # ------------------- ASR/WER evaluation -------------------
    def evaluate_wer_pair(self):
        if not self.transcripts_folder:
            QMessageBox.information(self, "Нет транскриптов", "Сначала выберите папку с транскриптами.")
            return
        if self.raw_sr <= 0 or len(self.raw_signal) == 0:
            QMessageBox.warning(self, "Нет файла", "Загрузите noisy/raw файл для оценки.")
            return
        sr = self.raw_sr
        transcript_path = self.find_transcript_file(self.ref_path if self.ref_path else self.raw_path)
        if not transcript_path:
            QMessageBox.warning(self, "Нет файла", "Не найден файл транскрипта (.txt) для текущего файла.")
            return
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                truth = f.read().strip()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть транскрипт: {e}")
            return
        if not self.pipeline:
            QMessageBox.information(self, "Pipeline пуст", "Сначала постройте pipeline.")
            return
        self.set_status("computing WER...")
        est = run_pipeline(self.raw_signal, self.raw_sr, self.pipeline)
        try:
            hyp = self.transcribe_audio(est, self.raw_sr)
        except Exception as e:
            self.set_status("готово")
            QMessageBox.critical(self, "ASR ошибка", str(e))
            return
        try:
            wer_val = self.compute_wer(truth, hyp)
        except Exception as e:
            self.set_status("готово")
            QMessageBox.critical(self, "WER ошибка", str(e))
            return
        QMessageBox.information(
            self,
            "WER оценка",
            f"Истинная транскрипция: {truth[:200]}...\n\n"
            f"Распознанная транскрипция: {hyp[:200]}...\n\n"
            f"Word Error Rate (WER) = {wer_val:.3f}")
        self.set_status("готово")

    def evaluate_wer_dataset(self):
        if not self.transcripts_folder:
            QMessageBox.information(self, "Нет транскриптов", "Сначала выберите папку с транскриптами.")
            return
        if not self.raw_folder or not self.ref_folder:
            QMessageBox.information(self, "Нет папок", "Сначала выберите папки raw и ref.")
            return
        if not self.pipeline:
            QMessageBox.information(self, "Pipeline пуст", "Сначала постройте pipeline.")
            return
        pairs = self.pair_dataset_files()
        if not pairs:
            QMessageBox.warning(self, "Нет пар", "Не найдено совпадающих файлов для оценки.")
            return
        wer_vals: List[float] = []
        skipped = 0
        total = len(pairs)
        self.set_status("evaluating dataset WER...")
        for idx, (raw_path, ref_path) in enumerate(pairs, start=1):
            trans_path = self.find_transcript_file(ref_path)
            if not trans_path:
                trans_path = self.find_transcript_file(raw_path)
            if not trans_path:
                skipped += 1
                if idx % 5 == 0 or idx == total:
                    self.set_status(f"WER dataset: {idx}/{total}")
                continue
            try:
                with open(trans_path, 'r', encoding='utf-8') as f:
                    truth = f.read().strip()
            except Exception:
                skipped += 1
                if idx % 5 == 0 or idx == total:
                    self.set_status(f"WER dataset: {idx}/{total}")
                continue
            raw, sr_raw = librosa.load(raw_path, sr=None, mono=True)
            raw = ensure_mono(raw)
            est = run_pipeline(raw, sr_raw, self.pipeline)
            try:
                hyp = self.transcribe_audio(est, sr_raw)
            except Exception:
                skipped += 1
                if idx % 5 == 0 or idx == total:
                    self.set_status(f"WER dataset: {idx}/{total}")
                continue
            try:
                wer_val = self.compute_wer(truth, hyp)
            except Exception:
                skipped += 1
                if idx % 5 == 0 or idx == total:
                    self.set_status(f"WER dataset: {idx}/{total}")
                continue
            wer_vals.append(wer_val)
            if idx % 5 == 0 or idx == total:
                self.set_status(f"WER dataset: {idx}/{total}")
        if not wer_vals:
            self.set_status("готово")
            QMessageBox.warning(self, "WER ошибка", "Не удалось вычислить WER для ни одной пары.")
            return
        mean_wer = float(np.mean(wer_vals))
        QMessageBox.information(
            self,
            "WER на датасете",
            f"Оценено пар: {len(wer_vals)}, пропущено: {skipped}.\nСредний WER = {mean_wer:.3f}")
        self.set_status("готово")

    # ------------------- Pipeline UI operations -------------------
    def add_selected_filter(self):
        items = self.available_list.selectedItems()
        if not items:
            return
        for item in items:
            cls = item.data(Qt.UserRole)
            if cls is not None:
                self.pipeline.append(cls())
        self.refresh_pipeline_list()

    def remove_pipeline_filter(self):
        row = self.pipeline_list.currentRow()
        if 0 <= row < len(self.pipeline):
            self.pipeline.pop(row)
            self.refresh_pipeline_list(row - 1)

    def move_up(self):
        row = self.pipeline_list.currentRow()
        if row > 0:
            self.pipeline[row - 1], self.pipeline[row] = self.pipeline[row], self.pipeline[row - 1]
            self.refresh_pipeline_list(row - 1)

    def move_down(self):
        row = self.pipeline_list.currentRow()
        if 0 <= row < len(self.pipeline) - 1:
            self.pipeline[row + 1], self.pipeline[row] = self.pipeline[row], self.pipeline[row + 1]
            self.refresh_pipeline_list(row + 1)

    def refresh_pipeline_list(self, current_row: Optional[int] = None):
        self.pipeline_list.clear()
        for i, filt in enumerate(self.pipeline):
            txt = f"{i + 1}. {filt.name}"
            self.pipeline_list.addItem(txt)
        if current_row is None:
            current_row = len(self.pipeline) - 1 if self.pipeline else -1
        if self.pipeline and current_row >= 0:
            self.pipeline_list.setCurrentRow(current_row)

    def clear_form(self):
        while self.params_form.rowCount() > 0:
            self.params_form.removeRow(0)

    def refresh_param_editor(self, row: int):
        self.clear_form()
        if row < 0 or row >= len(self.pipeline):
            return
        filt = self.pipeline[row]
        for pname, spec in filt.param_spec().items():
            ptype, value, pmin, pmax, pstep = spec
            if ptype is int:
                widget = QSpinBox()
                widget.setRange(int(pmin), int(pmax))
                widget.setSingleStep(int(max(1, pstep)))
                widget.setValue(int(value))
                widget.valueChanged.connect(lambda v, f=filt, n=pname: self.set_param(f, n, int(v)))
            elif ptype is str or pmin is None or pmax is None:
                from PyQt5.QtWidgets import QLineEdit
                widget = QLineEdit()
                widget.setText(str(value))
                widget.textChanged.connect(lambda text, f=filt, n=pname: self.set_param(f, n, text))
            else:
                widget = QDoubleSpinBox()
                widget.setRange(float(pmin), float(pmax))
                widget.setSingleStep(float(pstep))
                widget.setDecimals(4)
                widget.setValue(float(value))
                widget.valueChanged.connect(lambda v, f=filt, n=pname: self.set_param(f, n, float(v)))
            self.params_form.addRow(pname, widget)

    def set_param(self, filt: FilterBase, name: str, value: Any):
        filt.params[name] = value

    def set_status(self, message: str) -> None:
        self.lbl_status.setText(f"Status: {message}")
        QApplication.processEvents()

    # ------------------- Processing -------------------
    def get_eval_pair(self) -> Tuple[np.ndarray, np.ndarray, int]:
        if self.raw_sr <= 0 or self.ref_sr <= 0:
            raise ValueError("Нужно загрузить и raw, и reference.")
        ref = self.ref_signal
        if self.ref_sr != self.raw_sr:
            ref = resample_if_needed(self.ref_signal, self.ref_sr, self.raw_sr)
        return self.raw_signal, ref, self.raw_sr

    def apply_pipeline_clicked(self):
        """Apply the current pipeline to the raw signal.

        If a region has been selected via the span selector, only that
        segment of the raw (and reference) signal will be processed and
        compared.  Otherwise the full signals are used.  After
        processing, the metrics (if a reference is available) and
        latency are displayed, and the plots are refreshed.  During
        processing a status message is shown so the user knows that
        computation is in progress.
        """
        if self.raw_sr <= 0 or len(self.raw_signal) == 0:
            QMessageBox.information(self, "Нет входа", "Загрузите noisy/raw файл для обработки.")
            return
        if not self.pipeline:
            QMessageBox.information(self, "Пустой pipeline", "Добавьте хотя бы один фильтр.")
            return
        # Determine if a selection exists and compute indices for cropping
        sel_start = None
        sel_end = None
        if self.selection is not None and self.raw_sr > 0:
            start, end = self.selection
            s = min(start, end)
            e = max(start, end)
            duration = len(self.raw_signal) / self.raw_sr
            s = max(0.0, min(duration, s))
            e = max(0.0, min(duration, e))
            if e > s + 1e-3:
                sel_start = int(s * self.raw_sr)
                sel_end = int(e * self.raw_sr)
        # Prepare raw input (possibly cropped)
        if sel_start is not None and sel_end is not None:
            raw_in = self.raw_signal[sel_start:sel_end]
        else:
            raw_in = self.raw_signal
        # Run pipeline on selected segment
        self.set_status("running pipeline...")
        try:
            processed = run_pipeline(raw_in, self.raw_sr, self.pipeline)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка обработки: {e}")
            self.set_status("готово")
            return
        self.processed_signal = processed
        # Compute metrics on matching segment of reference if available
        if self.ref_sr > 0 and len(self.ref_signal) > 0:
            # Resample reference to match raw_sr if needed
            ref = self.ref_signal
            if self.ref_sr != self.raw_sr:
                ref = resample_if_needed(ref, self.ref_sr, self.raw_sr)
            if sel_start is not None and sel_end is not None:
                # Convert selection boundaries to indices of reference
                # Note: ref is already resampled to raw_sr
                ref_segment = ref[sel_start:sel_end]
            else:
                ref_segment = ref
            try:
                metrics = compute_metrics(ref_segment, processed, self.raw_sr)
                self.show_metrics(metrics, self.raw_sr)
            except Exception:
                # fallback: show no metrics if mismatch or error
                self.lbl_metrics.setText("Metrics: ошибка расчёта")
                self.lbl_latency.setText(f"Latency: {total_latency_ms(self.raw_sr, self.pipeline):.2f} ms")
        else:
            self.lbl_metrics.setText("Metrics: нет reference")
            self.lbl_latency.setText(f"Latency: {total_latency_ms(self.raw_sr, self.pipeline):.2f} ms")
        # Refresh plots (will highlight selection and plot PSD accordingly)
        self.redraw()
        self.set_status("готово")

    def show_metrics(self, metrics: Dict[str, float], sr: int):
        self.lbl_metrics.setText(
            f"Metrics: MSE={metrics['mse']:.6f} | SNR={metrics['snr']:.2f} dB | SegSNR={metrics['seg_snr']:.2f} dB | LSD={metrics['lsd']:.2f}"
        )
        self.lbl_latency.setText(f"Latency: {total_latency_ms(sr, self.pipeline):.2f} ms")

    def redraw(self):
        """
        Repaint the waveform and power spectrum plots.

        If a region has been selected via the span selector, the power
        spectrum will be computed only for the selected portion of the
        signal, and the selected region will be highlighted on the
        waveform plot.  Otherwise the full signals are used.
        """
        # Clear all axes
        self.canvas.ax_pan.clear()
        self.canvas.ax_wave.clear()
        self.canvas.ax_spec.clear()
        # Determine selection bounds in seconds
        sel_start: Optional[float] = None
        sel_end: Optional[float] = None
        if self.selection is not None and self.raw_sr > 0:
            start, end = self.selection
            # Clamp to [0, duration] range
            max_time = len(self.raw_signal) / self.raw_sr if len(self.raw_signal) > 0 else 0.0
            sel_start = max(0.0, min(start, end))
            sel_end = min(max_time, max(start, end))
            if sel_end <= sel_start + 1e-3:
                sel_start = None
                sel_end = None
        # Plot overview/panner
        if self.raw_sr > 0 and len(self.raw_signal) > 0:
            t_full = np.arange(len(self.raw_signal)) / self.raw_sr
            # Downsample overview if very long to speed up plotting
            if len(t_full) > 5000:
                step = max(1, len(t_full) // 5000)
                t_down = t_full[::step]
                y_down = self.raw_signal[::step]
            else:
                t_down = t_full
                y_down = self.raw_signal
            self.canvas.ax_pan.plot(t_down, y_down, color="gray", linewidth=0.8)
            # Highlight selection on overview
            if sel_start is not None and sel_end is not None:
                try:
                    self.canvas.ax_pan.axvspan(sel_start, sel_end, color='yellow', alpha=0.3)
                except Exception:
                    pass
            self.canvas.ax_pan.set_xlim(0, t_full[-1] if len(t_full) > 0 else 1.0)
            # Do not show y ticks for overview to save space
            self.canvas.ax_pan.set_yticks([])
        self.canvas.ax_pan.set_title("Overview")
        self.canvas.ax_pan.set_xlabel("Time, s")
        self.canvas.ax_pan.grid(True)

        # Plot waveform signals and highlight selection
        if self.raw_sr > 0 and len(self.raw_signal) > 0:
            t_raw = np.arange(len(self.raw_signal)) / self.raw_sr
            self.canvas.ax_wave.plot(t_raw, self.raw_signal, label="raw", alpha=0.7)
        if self.ref_sr > 0 and len(self.ref_signal) > 0:
            # Resample reference to match raw_sr if needed
            sr_ref = self.raw_sr or self.ref_sr
            ref = self.ref_signal if self.ref_sr == sr_ref else resample_if_needed(self.ref_signal, self.ref_sr, sr_ref)
            t_ref = np.arange(len(ref)) / sr_ref
            self.canvas.ax_wave.plot(t_ref, ref, label="reference", alpha=0.7)
        if self.raw_sr > 0 and len(self.processed_signal) > 0:
            t_proc = np.arange(len(self.processed_signal)) / self.raw_sr
            self.canvas.ax_wave.plot(t_proc, self.processed_signal, label="processed", alpha=0.8)
        # Highlight the selected region on the waveform
        if sel_start is not None and sel_end is not None:
            try:
                self.canvas.ax_wave.axvspan(sel_start, sel_end, color='yellow', alpha=0.3)
            except Exception:
                pass
        # Draw playback line
        try:
            self.playback_line = self.canvas.ax_wave.axvline(self.playback_x, color='red', linestyle='--')
        except Exception:
            pass
        self.canvas.ax_wave.set_title("Waveform")
        self.canvas.ax_wave.set_xlabel("Time, s")
        self.canvas.ax_wave.set_ylabel("Amplitude")
        self.canvas.ax_wave.grid(True)
        self.canvas.ax_wave.legend(loc="upper right")
        # Define helper to plot PSD for a (possibly selected) slice
        def plot_spec_slice(ax, y: np.ndarray, sr: int, label: str) -> None:
            if len(y) < 64:
                return
            f, pxx = welch(y, fs=sr, nperseg=min(2048, len(y)))
            ax.semilogy(f, pxx + 1e-12, label=label)
        # Plot PSD for raw, ref and processed signals with selection
        if self.raw_sr > 0 and len(self.raw_signal) > 0:
            y = self.raw_signal
            sr = self.raw_sr
            if sel_start is not None and sel_end is not None:
                start_idx = int(sel_start * sr)
                end_idx = int(sel_end * sr)
                start_idx = max(0, min(start_idx, len(y) - 1))
                end_idx = max(start_idx + 1, min(end_idx, len(y)))
                y_sel = y[start_idx:end_idx]
            else:
                y_sel = y
            plot_spec_slice(self.canvas.ax_spec, y_sel, sr, "raw")
        if self.ref_sr > 0 and len(self.ref_signal) > 0:
            sr_ref = self.raw_sr or self.ref_sr
            ref = self.ref_signal if self.ref_sr == sr_ref else resample_if_needed(self.ref_signal, self.ref_sr, sr_ref)
            if sel_start is not None and sel_end is not None:
                start_idx = int(sel_start * sr_ref)
                end_idx = int(sel_end * sr_ref)
                start_idx = max(0, min(start_idx, len(ref) - 1))
                end_idx = max(start_idx + 1, min(end_idx, len(ref)))
                ref_sel = ref[start_idx:end_idx]
            else:
                ref_sel = ref
            plot_spec_slice(self.canvas.ax_spec, ref_sel, sr_ref, "reference")
        if self.raw_sr > 0 and len(self.processed_signal) > 0:
            y = self.processed_signal
            sr = self.raw_sr
            if sel_start is not None and sel_end is not None:
                start_idx = int(sel_start * sr)
                end_idx = int(sel_end * sr)
                start_idx = max(0, min(start_idx, len(y) - 1))
                end_idx = max(start_idx + 1, min(end_idx, len(y)))
                y_sel = y[start_idx:end_idx]
            else:
                y_sel = y
            plot_spec_slice(self.canvas.ax_spec, y_sel, sr, "processed")
        self.canvas.ax_spec.set_title("Power spectrum")
        self.canvas.ax_spec.set_xlabel("Frequency, Hz")
        self.canvas.ax_spec.set_ylabel("PSD")
        self.canvas.ax_spec.grid(True)
        self.canvas.ax_spec.legend(loc="upper right")
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    # ------------------- Audio playback -------------------
    def write_temp_wav(self, y: np.ndarray, sr: int) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="tmp_audio_")
        os.close(fd)
        norm = np.clip(y / max(np.max(np.abs(y)), 1e-12), -1.0, 1.0)
        pcm = (norm * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        self.temp_files.append(path)
        return path

    def play_raw(self):
        if self.raw_sr <= 0 or len(self.raw_signal) == 0:
            QMessageBox.information(self, "Нет файла", "Raw файл не загружен.")
            return
        path = self.write_temp_wav(self.raw_signal, self.raw_sr)
        self.playback_x = 0.0
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.player.play()

    def play_processed(self):
        if self.raw_sr <= 0 or len(self.processed_signal) == 0:
            QMessageBox.information(self, "Нет результата", "Сначала примените pipeline.")
            return
        path = self.write_temp_wav(self.processed_signal, self.raw_sr)
        self.playback_x = 0.0
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.player.play()

    def pause_playback(self):
        state = self.player.state()
        if state == QMediaPlayer.PlayingState:
            self.player.pause()
            self.lbl_status.setText("Status: paused")
        elif state == QMediaPlayer.PausedState:
            self.player.play()
            self.lbl_status.setText("Status: playing")

    def play_reference(self):
        if self.ref_sr <= 0 or len(self.ref_signal) == 0:
            QMessageBox.information(self, "Нет файла", "Reference файл не загружен.")
            return
        path = self.write_temp_wav(self.ref_signal, self.ref_sr)
        self.playback_x = 0.0
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.player.play()

    # ------------------- Region selection (panner) -------------------
    def on_select_span(self, xmin: float, xmax: float) -> None:
        """
        Callback for the span selector on the waveform plot.

        Parameters
        ----------
        xmin, xmax : float
            The start and end times (in seconds) of the selected region.
        """
        # Reset selection if user clicks without dragging or selects an extremely
        # small region.  Otherwise store the sorted interval.
        if abs(xmax - xmin) < 1e-3:
            self.selection = None
        else:
            self.selection = (float(min(xmin, xmax)), float(max(xmin, xmax)))
        # Redraw with the new selection highlighted and spectrum updated
        self.redraw()

    def on_select_pan(self, xmin: float, xmax: float) -> None:
        """
        Callback for the span selector on the panner (overview) plot.

        This behaves similarly to :meth:`on_select_span`, but is
        triggered when the user drags on the overview axis.  It sets
        :attr:`selection` to the chosen interval (in seconds) and
        triggers a redraw.  If the selection is too small (less than
        1e-3 seconds) the selection is cleared.

        Parameters
        ----------
        xmin, xmax : float
            The start and end times (in seconds) of the selected region
            on the overview plot.
        """
        if abs(xmax - xmin) < 1e-3:
            self.selection = None
        else:
            self.selection = (float(min(xmin, xmax)), float(max(xmin, xmax)))
        # Redraw both panner and waveform plots with the new selection
        self.redraw()

    def on_position_changed(self, position_ms: int) -> None:
        self.playback_x = position_ms / 1000.0
        try:
            if self.playback_line is not None:
                self.playback_line.set_xdata([self.playback_x, self.playback_x])
            self.canvas.draw_idle()
        except Exception:
            pass

    # ------------------- Optimization & search -------------------
    def optimize_current_pipeline(self):
        try:
            raw, ref, sr = self.get_eval_pair()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", str(e))
            return
        if not self.pipeline:
            QMessageBox.information(self, "Пусто", "Pipeline пуст.")
            return
        search_lists = [f.search_space() for f in self.pipeline]
        total = 1
        for s in search_lists:
            total *= max(1, len(s))
        max_eval = self.spin_maxeval.value()
        if total > max_eval:
            QMessageBox.information(
                self,
                "Слишком большой перебор",
                f"Для текущего pipeline получится {total} комбинаций, будет оценено только первые {max_eval}.",
            )
        best_score = -1e18
        best_pipeline = None
        best_metrics = None
        count = 0
        self.set_status("optimizing parameters...")
        for params_combo in itertools.product(*search_lists):
            candidate = [f.__class__(**params) for f, params in zip(self.pipeline, params_combo)]
            est = run_pipeline(raw, sr, candidate)
            metrics = compute_metrics(ref, est, sr)
            score = composite_score(metrics)
            if score > best_score:
                best_score = score
                best_pipeline = candidate
                best_metrics = metrics
                self.processed_signal = est
            count += 1
            if count % 10 == 0:
                self.set_status(f"Optimizing: {count}/{min(total, max_eval)} combos")
            if count >= max_eval:
                break
        if best_pipeline is not None:
            self.pipeline = best_pipeline
            self.refresh_pipeline_list()
            self.refresh_param_editor(self.pipeline_list.currentRow())
            if best_metrics is not None:
                self.show_metrics(best_metrics, sr)
                self.redraw()
            QMessageBox.information(self, "Готово", f"Перебор завершен. Проверено {count} комбинаций.")
        self.set_status("готово")

    def selected_filter_classes(self) -> List[Any]:
        classes: List[Any] = []
        for item in self.available_list.selectedItems():
            cls = item.data(Qt.UserRole)
            if cls is not None:
                classes.append(cls)
        return classes

    def search_best_pipelines(self):
        try:
            raw, ref, sr = self.get_eval_pair()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", str(e))
            return
        classes = self.selected_filter_classes()
        if not classes:
            QMessageBox.information(self, "Нет выбора", "Выберите 1+ фильтров в списке слева.")
            return
        max_depth = self.spin_depth.value()
        top_k = self.spin_topk.value()
        max_eval = self.spin_maxeval.value()
        results: List[Dict[str, Any]] = []
        eval_count = 0
        self.set_status("searching best pipelines...")
        for depth in range(1, max_depth + 1):
            for cls_seq in itertools.product(classes, repeat=depth):
                space_lists = [cls().search_space() for cls in cls_seq]
                for param_combo in itertools.product(*space_lists):
                    pipeline = [cls(**params) for cls, params in zip(cls_seq, param_combo)]
                    est = run_pipeline(raw, sr, pipeline)
                    metrics = compute_metrics(ref, est, sr)
                    score = composite_score(metrics)
                    result = {
                        "pipeline": pipeline,
                        "metrics": metrics,
                        "score": score,
                        "desc": " -> ".join(f.name for f in pipeline),
                    }
                    results.append(result)
                    eval_count += 1
                    if eval_count % 20 == 0:
                        self.set_status(f"Searching: {eval_count}/{max_eval} combos")
                    if eval_count >= max_eval:
                        break
                if eval_count >= max_eval:
                    break
            if eval_count >= max_eval:
                break
        results.sort(key=lambda x: x["score"], reverse=True)
        self.search_results = results[:top_k]
        self.results_list.clear()
        for idx, res in enumerate(self.search_results, start=1):
            m = res["metrics"]
            txt = (
                f"{idx}. {res['desc']} | "
                f"SNR={m['snr']:.2f} | SegSNR={m['seg_snr']:.2f} | LSD={m['lsd']:.2f} | MSE={m['mse']:.6f}"
            )
            self.results_list.addItem(txt)
        if self.search_results:
            self.results_list.setCurrentRow(0)
            best = self.search_results[0]
            self.processed_signal = run_pipeline(raw, sr, best["pipeline"])
            self.show_metrics(best["metrics"], sr)
            self.redraw()
        QMessageBox.information(self, "Поиск завершен", f"Оценено {eval_count} комбинаций.")
        self.set_status("готово")

    def apply_selected_search_result(self):
        row = self.results_list.currentRow()
        if row < 0 or row >= len(self.search_results):
            return
        res = self.search_results[row]
        self.pipeline = [f.clone() for f in res["pipeline"]]
        self.refresh_pipeline_list(0 if self.pipeline else -1)
        self.refresh_param_editor(self.pipeline_list.currentRow())
        try:
            raw, ref, sr = self.get_eval_pair()
            self.processed_signal = run_pipeline(raw, sr, self.pipeline)
            self.show_metrics(res["metrics"], sr)
            self.redraw()
        except Exception:
            pass

    def export_results(self):
        if not self.search_results:
            QMessageBox.information(self, "Нет результатов", "Сначала запустите поиск.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результаты",
            "results.json",
            "JSON (*.json);;CSV (*.csv)"
        )
        if not path:
            return
        if path.lower().endswith(".csv"):
            import csv
            keys = ["desc", "snr", "seg_snr", "lsd", "mse"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for res in self.search_results:
                    m = res["metrics"]
                    writer.writerow([
                        res["desc"],
                        f"{m['snr']:.4f}",
                        f"{m['seg_snr']:.4f}",
                        f"{m['lsd']:.4f}",
                        f"{m['mse']:.8f}",
                    ])
        else:
            out: List[Dict[str, Any]] = []
            for res in self.search_results:
                m = res["metrics"]
                out.append({
                    "pipeline": res["desc"],
                    "snr": m["snr"],
                    "seg_snr": m["seg_snr"],
                    "lsd": m["lsd"],
                    "mse": m["mse"],
                })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "Сохранено", f"Результаты сохранены в {os.path.basename(path)}")

    # ------------------- Dataset evaluation -------------------
    def find_audio_files(self, folder: str) -> List[str]:
        out: List[str] = []
        if not folder:
            return out
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}:
                out.append(path)
        return sorted(out)

    def pair_dataset_files(self) -> List[Tuple[str, str]]:
        raw_files = self.find_audio_files(self.raw_folder)
        ref_files = self.find_audio_files(self.ref_folder)
        raw_map = {normalize_pairing_key(p): p for p in raw_files}
        ref_map = {normalize_pairing_key(p): p for p in ref_files}
        pairs: List[Tuple[str, str]] = []
        for key, raw_path in raw_map.items():
            if key in ref_map:
                pairs.append((raw_path, ref_map[key]))
        return sorted(pairs)

    def evaluate_dataset_folders(self):
        if not self.raw_folder or not self.ref_folder:
            QMessageBox.information(self, "Нет папок", "Сначала выберите папки raw и ref.")
            return
        if not self.pipeline:
            QMessageBox.information(self, "Пустой pipeline", "Добавьте фильтры в pipeline.")
            return
        pairs = self.pair_dataset_files()
        if not pairs:
            QMessageBox.warning(
                self,
                "Нет пар",
                "Не найдено совпадающих файлов. Поиск делает pairing по имени файла с удалением хвостовых '_' и номеров в скобках.",
            )
            return
        self.set_status("evaluating dataset...")
        metrics_list: List[Dict[str, float]] = []
        total_pairs = len(pairs)
        for idx, (raw_path, ref_path) in enumerate(pairs, start=1):
            raw, sr_raw = librosa.load(raw_path, sr=None, mono=True)
            ref, sr_ref = librosa.load(ref_path, sr=None, mono=True)
            raw = ensure_mono(raw)
            ref = ensure_mono(ref)
            if sr_ref != sr_raw:
                ref = resample_if_needed(ref, sr_ref, sr_raw)
            est = run_pipeline(raw, sr_raw, self.pipeline)
            metrics_list.append(compute_metrics(ref, est, sr_raw))
            if idx % 5 == 0 or idx == total_pairs:
                self.set_status(f"Dataset evaluation: {idx}/{total_pairs}")
        mean_metrics = {
            key: float(np.mean([m[key] for m in metrics_list]))
            for key in ["mse", "snr", "seg_snr", "lsd"]
        }
        self.lbl_metrics.setText(
            f"Batch metrics ({len(pairs)} pairs): "
            f"MSE={mean_metrics['mse']:.6f} | "
            f"SNR={mean_metrics['snr']:.2f} dB | "
            f"SegSNR={mean_metrics['seg_snr']:.2f} dB | "
            f"LSD={mean_metrics['lsd']:.2f}"
        )
        QMessageBox.information(self, "Batch evaluation", f"Обработано {len(pairs)} пар файлов.")
        self.set_status("готово")

    # ------------------- Help dialog -------------------
    def show_help(self):
        text = (
            "<b>Основные элементы интерфейса</b><br><br>"
            "<b>Raw / Reference:</b> загрузите аудиофайлы. Raw — шумовой (или необработанный) сигнал, Reference — эталонный сигнал для оценки.<br>"
            "<b>Pipeline:</b> слева выбирайте фильтры, добавляйте их в последовательность (pipeline). Порядок применения важен.<br>"
            "<b>Кнопки Raw / Processed / Reference / Пауза:</b> позволяют прослушивать соответствующие аудиосигналы. Кнопка 'Пауза' ставит воспроизведение на паузу или возобновляет его.<br>"
            "<b>Применить pipeline:</b> применяет текущий pipeline к raw‑сигналу. Если загружен reference, выводятся метрики (MSE, SNR, сегментальный SNR, LSD).<br>"
            "<b>Перебор параметров текущего pipeline:</b> перебирает параметры выбранных фильтров в pipeline, выбирая комбинацию с наилучшей метрикой.<br>"
            "<b>Brute‑force поиск комбинаций:</b> ищет лучшие сочетания фильтров заданной длины из выбранных в списке.<br>"
            "<b>Сохранить/Открыть pipeline:</b> сохранение и загрузка вашей последовательности фильтров (JSON).<br>"
            "<b>ASR/WER оценка:</b> требует папку с транскриптами (.txt). Выполняет распознавание речи и вычисляет Word Error Rate (WER) для текущей записи или датасета.<br>"
            "<b>Папка transcripts:</b> выберите папку, где хранятся текстовые транскрипции (имя файла без расширения должно совпадать с аудиофайлом).<br><br>"
            "<b>Параметры фильтров</b><br>"
            "<b>Low-pass/High-pass/Band-pass:</b> задаются граничные частоты и порядок фильтра.<br>"
            "<b>Notch 50/60Hz:</b> freq — центральная частота режекторного фильтра, q — добротность.<br>"
            "<b>Moving average:</b> window — размер скользящего окна.<br>"
            "<b>Median:</b> kernel_size — размер ядра медианного фильтра (нечётное).<br>"
            "<b>Wiener:</b> size — размер окна фильтра Винера.<br>"
            "<b>Temporal difference:</b> alpha — коэффициент подавления предыдущего отсчёта.<br>"
            "<b>VAD noise gate:</b> threshold_mult — множитель для порога детекции речи; silence_gain — усиление фона в паузах; frame_ms и hop_ms — размер и шаг кадров.<br>"
            "<b>Spectral subtraction:</b> strength — степень вычитания шума; floor_ratio — минимальный уровень шума; n_fft — размер FFT; hop_ms — шаг; vad_threshold_mult и noise_percentile — параметры VAD для оценки шумового профиля.<br>"
            "<b>NLMS/RLS:</b> length — длина фильтра; mu/lam/delta — параметры адаптации.<br>"
            "<b>MMSE (approx.):</b> статистическая оценка спектральной амплитуды. alpha — сглаживание априорной SNR; floor_ratio — минимальный уровень подавления; n_fft/hop_ms — параметры анализа; vad_threshold_mult и noise_percentile — параметры VAD.<br>"
            "<b>Wavelet denoise:</b> вейвлет‑шринк. wavelet — семейство; level — уровень разложения; threshold — множитель порога. Требуется pywt.<br>"
            "<b>Subspace (SVD):</b> метод субпространств: оставляет несколько доминирующих сингулярных векторов. n_components — количество компонент; n_fft/hop_ms — параметры STFT.<br>"
            "<b>NMF:</b> неотрицательная матричная факторизация. n_components — число компонент; speech_ratio — доля компонент, считающихся речью; n_fft/hop_ms — параметры STFT.<br>"
            "<b>Kalman:</b> простой фильтр Калмана. process_noise и measurement_noise задают дисперсии шумов.<br>"
            "<b>RNNoise/Demucs:</b> заглушки для глубинных моделей. При наличии соответствующих библиотек будет использован реальный алгоритм; иначе применяется fallback.<br><br>"
            "<b>Overview (паннер):</b> небольшая панель над графиком waveform показывает весь raw‑сигнал.  Перетяните мышью по панели, чтобы выделить интересующий участок.  Выделенный диапазон подсветится жёлтым и будет использован для вычисления спектра и применения pipeline.  Изменить или переместить диапазон можно, выделив новую область.<br><br>"
            "Приложение построено по модульному принципу.  Все фильтры определены в пакете backend/filters и автоматически загружаются при запуске.  Это упрощает расширение: чтобы добавить новый фильтр, поместите новый класс, наследующий FilterBase, в backend/filters.  Бизнес‑логика и UI разделены, поэтому ядро можно переиспользовать в других интерфейсах или микросервисах.")
        QMessageBox.information(self, "Справка", text)

    def closeEvent(self, event):
        try:
            for path in self.temp_files:
                try:
                    os.remove(path)
                except Exception:
                    pass
        finally:
            event.accept()


def main():
    app = QApplication(sys.argv)
    window = AudioFilterApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()