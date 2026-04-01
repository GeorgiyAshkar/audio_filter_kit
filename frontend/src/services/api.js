import axios from 'axios'

export const api = axios.create({ baseURL: import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8000' })
