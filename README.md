# Audio Filter Kit (Web)

Веб-версия приложения для фильтрации аудио с сохранением ключевого функционала исходного desktop-инструмента:

- Pipeline из фильтров (модульно).
- Обработка raw/reference.
- Метрики качества (MSE/SNR/SegSNR/LSD).
- Графики Overview (panner), Waveform, Power Spectrum.
- Сохранение/загрузка pipeline.
- Прослушивание processed аудио.

## Архитектура

Приложение разделено на микросервисно-модульные слои:

- `app.py` — web entrypoint и HTTP API.
- `backend/audio_io.py` — декодирование/кодирование аудио.
- `backend/filters.py` — библиотека фильтров и pipeline engine.
- `backend/metrics.py` — метрики и scoring.
- `templates/`, `static/` — React UI.

## Быстрый запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

После запуска откройте: http://127.0.0.1:8000

## Проверка panner

1. Загрузите raw/reference.
2. На графике **Overview (panner)** выделите участок через zoom/range slider.
3. Нажмите **Применить pipeline**.
4. Убедитесь, что пересчитываются спектр и обработка только для выбранного диапазона.
