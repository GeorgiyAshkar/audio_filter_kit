const { useEffect, useMemo, useState } = React;

function App() {
  const [filters, setFilters] = useState({});
  const [availableSelected, setAvailableSelected] = useState([]);
  const [pipeline, setPipeline] = useState([]);
  const [rawFile, setRawFile] = useState(null);
  const [refFile, setRefFile] = useState(null);
  const [result, setResult] = useState(null);
  const [selection, setSelection] = useState(null);

  useEffect(() => { fetch('/api/filters').then(r => r.json()).then(d => setFilters(d.filters)); }, []);
  useEffect(() => { if (result) drawPlots(result); }, [result]);

  const addFilter = () => {
    const additions = availableSelected.map(name => ({ name, params: Object.fromEntries(Object.entries(filters[name]).map(([k,v]) => [k, v[0]])) }));
    setPipeline([...pipeline, ...additions]);
  };

  const applyPipeline = async () => {
    if (!rawFile) return alert('Загрузите raw файл');
    const form = new FormData();
    form.append('raw', rawFile);
    if (refFile) form.append('reference', refFile);
    form.append('pipeline', JSON.stringify(pipeline));
    form.append('selection', JSON.stringify(selection));
    const r = await fetch('/api/process', { method: 'POST', body: form });
    const d = await r.json();
    if (d.error) return alert(d.error);
    setResult(d);
  };

  const savePipeline = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(pipeline, null, 2)], {type:'application/json'}));
    a.download = 'pipeline.json'; a.click();
  };

  const loadPipeline = (f) => {
    const reader = new FileReader();
    reader.onload = () => setPipeline(JSON.parse(reader.result));
    reader.readAsText(f);
  };

  const playProcessed = () => {
    if (!result?.processedWav) return;
    const audio = new Audio('data:audio/wav;base64,' + result.processedWav);
    audio.play();
  };

  return <div className="app">
    <div className="topbar panel">
      <input type="file" accept="audio/*" onChange={e=>setRawFile(e.target.files[0])}/>
      <input type="file" accept="audio/*" onChange={e=>setRefFile(e.target.files[0])}/>
      <button onClick={savePipeline}>Сохранить pipeline</button>
      <input type="file" accept="application/json" onChange={e=>loadPipeline(e.target.files[0])} />
    </div>
    <div className="grid">
      <div className="panel left">
        <h4>Доступные фильтры</h4>
        <select multiple size="10" onChange={e => setAvailableSelected([...e.target.selectedOptions].map(o=>o.value))}>
          {Object.keys(filters).map(f => <option key={f} value={f}>{f}</option>)}
        </select>
        <button onClick={addFilter}>Добавить →</button>
        <h4>Pipeline</h4>
        {pipeline.map((p, idx)=><div key={idx} className="panel">
          <b>{idx+1}. {p.name}</b>
          {Object.entries(p.params).map(([k,v]) => <div key={k}><label>{k}</label><input value={v} onChange={e=>{
            const n = [...pipeline]; n[idx].params[k] = Number(e.target.value); setPipeline(n);
          }}/></div>)}
          <button onClick={()=>setPipeline(pipeline.filter((_,i)=>i!==idx))}>Удалить</button>
        </div>)}
      </div>
      <div className="panel middle">
        <h4>Применение и прослушивание</h4>
        <button onClick={applyPipeline}>Применить pipeline</button>
        <button onClick={playProcessed}>Processed</button>
        {result?.metrics && <div>
          <p>MSE: {result.metrics.mse.toFixed(6)}</p>
          <p>SNR: {result.metrics.snr.toFixed(2)} dB</p>
          <p>SegSNR: {result.metrics.seg_snr.toFixed(2)} dB</p>
          <p>LSD: {result.metrics.lsd.toFixed(3)}</p>
          <p>Latency: {result.latencyMs.toFixed(2)} ms</p>
        </div>}
        <p className='small'>Panner: выделяйте диапазон в верхнем overview графике через zoom/range slider.</p>
      </div>
      <div className="panel right">
        <div id="overview" className="plot"></div>
        <div id="wave" className="plot"></div>
        <div id="spec" className="plot"></div>
      </div>
    </div>
  </div>;

  function drawPlots(d) {
    const sr = d.sr;
    const tx = d.rawSignal.map((_,i)=>i/sr);
    Plotly.newPlot('overview', [{x:tx, y:d.rawSignal, type:'scatter', name:'raw'}], {title:'Overview (panner)', xaxis:{rangeslider:{visible:true}}});
    const wData = [{x:tx, y:d.rawSignal, name:'raw', type:'scatter'}];
    if (d.referenceSignal?.length) wData.push({x:d.referenceSignal.map((_,i)=>i/sr), y:d.referenceSignal, name:'reference', type:'scatter'});
    if (d.processedSignal?.length) wData.push({x:d.processedSignal.map((_,i)=>i/sr), y:d.processedSignal, name:'processed', type:'scatter'});
    Plotly.newPlot('wave', wData, {title:'Waveform'});
    const s=[];
    if (d.spectra.raw.freq.length) s.push({x:d.spectra.raw.freq, y:d.spectra.raw.pow, name:'raw', type:'scatter'});
    if (d.spectra.reference.freq.length) s.push({x:d.spectra.reference.freq, y:d.spectra.reference.pow, name:'reference', type:'scatter'});
    if (d.spectra.processed.freq.length) s.push({x:d.spectra.processed.freq, y:d.spectra.processed.pow, name:'processed', type:'scatter'});
    Plotly.newPlot('spec', s, {title:'Power spectrum', yaxis:{type:'log'}});

    const ov = document.getElementById('overview');
    ov.on('plotly_relayout', (ev) => {
      if (ev['xaxis.range[0]'] !== undefined && ev['xaxis.range[1]'] !== undefined) {
        setSelection({start: Number(ev['xaxis.range[0]']), end: Number(ev['xaxis.range[1]'])});
      }
    });
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
