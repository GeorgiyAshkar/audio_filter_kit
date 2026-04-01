export default function UploadPanel({ onSetIds }) {
  return <div><h3>Upload Panel</h3><button onClick={() => onSetIds({ rawId: 'raw-demo', referenceId: 'ref-demo' })}>Load demo ids</button></div>
}
