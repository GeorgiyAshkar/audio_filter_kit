# Audio Filter Kit — Web TypeScript Migration

Приложение переведено с desktop Python GUI в web application на TypeScript с сохранением ключевого сценария:

- загрузка аудио;
- конфигурация pipeline фильтров;
- применение pipeline;
- просмотр panner/overview + waveform + spectrum графиков;
- прослушивание обработанного результата;
- расчет метрик качества и задержки.

## Микросервисная модульная архитектура

- `server/index.ts` — API gateway (Fastify) для health/process endpoint.
- `server/filters.ts` — отдельный модуль обработки pipeline (может быть вынесен в отдельный DSP сервис).
- `server/metrics.ts` — отдельный модуль вычисления метрик.
- `src/services/audio.ts` — клиентский API слой и кодирование/декодирование аудио.
- `src/components/*` — UI компоненты React.
- `src/domain/types.ts` — shared contract между frontend/backend.

## Запуск

1. `npm install`
2. Запуск API: `npm run server`
3. Запуск web UI: `npm run dev`

## Проверка panner

Верхний график (overview) содержит range slider и обрабатывает `onRelayout` для задания `[start, end]`, который затем синхронизируется с основным waveform графиком и спектром.
