import { useEffect, useState } from 'react'
import { api } from '../services/api'

export function useFilters() {
  const [filters, setFilters] = useState([])
  useEffect(() => {
    api.get('/filters/filters').then((r) => setFilters(r.data)).catch(() => setFilters([]))
  }, [])
  return filters
}
