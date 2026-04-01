# Audio Filter Kit (React + TypeScript)

Единая точка запуска: **`python3 app.py`**.

`app.py` теперь запускает **React (Vite)**, без Flask.

## Что делает `python3 app.py`
1. Проверяет наличие `npm`.
2. Устанавливает зависимости фронтенда (`npm install`), если нет `node_modules`.
3. Запускает React-приложение (`npm run dev -- --host 0.0.0.0 --port 5173`).

## Запуск

```bash
python3 app.py
```

После старта откройте:
- `http://127.0.0.1:5173`

## Альтернативно вручную
```bash
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

## Зависимости
### Node.js
- Node.js 18+
- npm

### Python
- стандартная библиотека Python 3.10+

### Optional DSP (если используете Python backend отдельно)
```bash
pip install -r requirements.txt
```
