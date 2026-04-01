import type { FilterConfig, FilterType } from '../domain/types';

const filterDefs: Record<FilterType, { name: string; defaults: Record<string, number> }> = {
  gain: { name: 'Gain', defaults: { gain: 1.2 } },
  normalize: { name: 'Normalize', defaults: { targetPeak: 0.9 } },
  moving_average: { name: 'Moving Average', defaults: { window: 5 } },
  high_pass: { name: 'High Pass', defaults: { alpha: 0.95 } },
  low_pass: { name: 'Low Pass', defaults: { alpha: 0.15 } }
};

export function createFilter(type: FilterType): FilterConfig {
  const def = filterDefs[type];
  return { id: crypto.randomUUID(), type, name: def.name, params: { ...def.defaults } };
}

export function FilterPanel({
  pipeline,
  onAdd,
  onRemove,
  onUpdate
}: {
  pipeline: FilterConfig[];
  onAdd: (type: FilterType) => void;
  onRemove: (id: string) => void;
  onUpdate: (id: string, key: string, value: number) => void;
}) {
  return (
    <div className="panel">
      <h3>Доступные фильтры</h3>
      <div className="btn-grid">
        {(Object.keys(filterDefs) as FilterType[]).map((type) => (
          <button key={type} onClick={() => onAdd(type)}>{filterDefs[type].name}</button>
        ))}
      </div>

      <h3>Pipeline</h3>
      {pipeline.map((filter) => (
        <div key={filter.id} className="filter-card">
          <div className="row between">
            <strong>{filter.name}</strong>
            <button onClick={() => onRemove(filter.id)}>Удалить</button>
          </div>
          {Object.entries(filter.params).map(([key, value]) => (
            <label key={key} className="row between">
              <span>{key}</span>
              <input type="number" value={value} step="0.01" onChange={(e) => onUpdate(filter.id, key, Number(e.target.value))} />
            </label>
          ))}
        </div>
      ))}
    </div>
  );
}
