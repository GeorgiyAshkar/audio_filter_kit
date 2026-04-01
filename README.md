# DAS Speech Filter Explorer

Монорепозиторий с микросервисной архитектурой и React SPA для обработки DAS-аудио.

## Реализованные сервисы

- `Gateway Service` (`services/gateway`) — единая точка входа с проксированием запросов.
- `Audio Storage Service` (`services/audio_storage`) — загрузка raw/reference и выдача аудио.
- `Filter Registry Service` (`services/filter_registry`) — каталог фильтров и валидация pipeline.
- `Processing Service` (`services/processing`) — plug-in фильтры и расчёт метрик (MSE, SNR, Segmental SNR, LSD).
- `Optimization Service` (`services/optimization`) — API для optimize/search задач.
- `Dataset Evaluation Service` (`services/dataset_evaluation`) — пакетная оценка набора.
- `Speech Recognition Service` (`services/speech_recognition`) — контракт WER-оценки.
- `Config Service` (`services/config_service`) — сохранение/загрузка pipeline.

## Frontend (React)

`frontend/` содержит SPA c требуемыми панелями:
- Upload Panel
- Pipeline Builder
- Segment selector (Panner/Waveform placeholder)
- Metrics Display
- Optimization Panel
- Dataset Evaluation Panel
- Transcript Folder Selector
- Help Dialog

Логика запросов вынесена в кастомные hooks (`useFilters`, `useProcessSegment`) и слой `services/api.js`.

## Запуск

```bash
pip install -r requirements.txt
pytest
docker compose up
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Самопроверка соответствия ТЗ

- [x] SPA на React с раздельной логикой (hooks/services) и UI-компонентами.
- [x] Все основные микросервисы из ТЗ присутствуют.
- [x] REST-контракты из ТЗ добавлены как endpoint-и.
- [x] Processing реализован через расширяемый plug-in registry фильтров.
- [x] Асинхронные long-running задачи отражены через task status API (готово для брокера/WS интеграции).
- [x] Добавлены базовые тесты межсервисных контрактов.

## Что нужно для production

1. Добавить JWT-аутентификацию и rate-limiting в Gateway.
2. Подключить брокер (RabbitMQ/Kafka) и WebSocket/SSE для прогресса задач.
3. Заменить in-memory хранилища на Postgres + S3-совместимое storage.
4. Интегрировать реальный ASR (Vosk/Whisper) и вычисление WER по транскриптам.
5. Доработать waveform/spectrum визуализацию (wavesurfer + FFT/Chart.js).
