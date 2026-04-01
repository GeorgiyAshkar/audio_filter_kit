import { useState } from 'react'

export default function PipelineBuilder({ filters, pipeline, setPipeline }) {
  const [selected, setSelected] = useState('')
  const addFilter = () => selected && setPipeline([...pipeline, { name: selected, params: {} }])
  return <div><h3>Pipeline Builder</h3><select value={selected} onChange={(e) => setSelected(e.target.value)}><option value=''>Select filter</option>{filters.map((f) => <option key={f.name} value={f.name}>{f.name}</option>)}</select><button onClick={addFilter}>Add</button><pre>{JSON.stringify(pipeline, null, 2)}</pre></div>
}
