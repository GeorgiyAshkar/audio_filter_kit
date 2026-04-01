import { useState } from 'react'
import UploadPanel from './components/UploadPanel'
import PipelineBuilder from './components/PipelineBuilder'
import MetricsDisplay from './components/MetricsDisplay'
import OptimizationPanel from './components/OptimizationPanel'
import DatasetEvalPanel from './components/DatasetEvalPanel'
import TranscriptFolderSelector from './components/TranscriptFolderSelector'
import HelpDialog from './components/HelpDialog'
import { useFilters } from './hooks/useFilters'
import { useProcessSegment } from './hooks/useProcessSegment'

export default function App() {
  const filters = useFilters()
  const [ids, setIds] = useState({ rawId: '', referenceId: '' })
  const [pipeline, setPipeline] = useState([])
  const [segment, setSegment] = useState({ start: 0, end: 2 })
  const { process, result, loading } = useProcessSegment()

  return (
    <main>
      <h1>DAS Speech Filter Explorer</h1>
      <UploadPanel onSetIds={setIds} />
      <PipelineBuilder filters={filters} pipeline={pipeline} setPipeline={setPipeline} />
      <section>
        <h3>Panner & Waveform</h3>
        <label>Start <input type='number' value={segment.start} onChange={(e) => setSegment({ ...segment, start: Number(e.target.value) })} /></label>
        <label>End <input type='number' value={segment.end} onChange={(e) => setSegment({ ...segment, end: Number(e.target.value) })} /></label>
      </section>
      <button disabled={loading || !ids.rawId} onClick={() => process({ ...ids, pipeline, segment })}>Apply</button>
      <MetricsDisplay metrics={result?.metrics} />
      <OptimizationPanel />
      <DatasetEvalPanel />
      <TranscriptFolderSelector />
      <HelpDialog />
    </main>
  )
}
