export default function MetricsDisplay({ metrics }) {
  if (!metrics) return <div><h3>Metrics</h3><p>No metrics yet</p></div>
  return <div><h3>Metrics</h3><ul>{Object.entries(metrics).map(([k,v]) => <li key={k}>{k}: {String(v)}</li>)}</ul></div>
}
