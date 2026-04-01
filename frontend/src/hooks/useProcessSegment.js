import { useState } from 'react'
import { api } from '../services/api'

export function useProcessSegment() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const process = async (payload) => {
    setLoading(true)
    const { data } = await api.post('/processing/process', payload)
    setResult(data)
    setLoading(false)
  }

  return { process, result, loading }
}
