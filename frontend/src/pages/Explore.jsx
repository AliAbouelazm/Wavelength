import { useEffect, useState } from 'react';
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchExplore } from '../api/client';
import styles from '../styles/explore.module.css';

const MOOD_COLORS = {
  Happy: '#92400e',
  Energetic: '#991b1b',
  Melancholic: '#4c1d95',
  Focused: '#064e3b',
  Calm: '#1e3a8a',
  Intense: '#374151',
  Neutral: '#9ca3af',
};

const MOODS = ['Happy', 'Energetic', 'Melancholic', 'Focused', 'Calm', 'Intense', 'Neutral'];

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div className={styles.tooltip}>
      <p className={styles.tooltipName}>{d.track_name}</p>
      <p className={styles.tooltipArtist}>{d.artists}</p>
      <p className={styles.tooltipMood}>{d.mood}</p>
    </div>
  );
}

export default function Explore() {
  const [tracks, setTracks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tempoMin, setTempoMin] = useState(0);
  const [tempoMax, setTempoMax] = useState(1);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    fetchExplore()
      .then(setTracks)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const filtered = tracks.filter(
    (t) => t.tempo_norm >= tempoMin && t.tempo_norm <= tempoMax
  );

  const byMood = MOODS.reduce((acc, m) => {
    acc[m] = filtered.filter((t) => t.mood === m);
    return acc;
  }, {});

  return (
    <main className={styles.page}>
      <div className={styles.header}>
        <h1 className={styles.heading}>Feature space</h1>
        <p className={styles.subheading}>
          Every track in the dataset, plotted by energy and mood tone (valence). Color shows predicted mood.
        </p>
      </div>

      {loading && <p className={styles.loading}>Loading tracks...</p>}
      {error && <p className={styles.error}>{error}</p>}

      {!loading && !error && (
        <>
          <div className={styles.tempoFilter}>
            <label className={styles.filterLabel}>
              Tempo range: {Math.round(tempoMin * 250)} bpm to {Math.round(tempoMax * 250)} bpm
            </label>
            <div className={styles.rangeRow}>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={tempoMin}
                onChange={(e) => setTempoMin(parseFloat(e.target.value))}
                className={styles.slider}
              />
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={tempoMax}
                onChange={(e) => setTempoMax(parseFloat(e.target.value))}
                className={styles.slider}
              />
            </div>
          </div>

          <div className={styles.legend}>
            {MOODS.map((m) => (
              <span key={m} className={styles.legendItem}>
                <span
                  className={styles.legendDot}
                  style={{ background: MOOD_COLORS[m] }}
                />
                {m}
              </span>
            ))}
          </div>

          <div className={styles.chartWrap}>
            <ResponsiveContainer width="100%" height={480}>
              <ScatterChart margin={{ top: 8, right: 24, bottom: 24, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  type="number"
                  dataKey="energy"
                  name="Energy"
                  domain={[0, 1]}
                  label={{ value: 'Energy', position: 'insideBottom', offset: -12, fontSize: 12, fill: 'var(--text-secondary)' }}
                  tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  type="number"
                  dataKey="valence"
                  name="Valence"
                  domain={[0, 1]}
                  label={{ value: 'Valence', angle: -90, position: 'insideLeft', offset: 12, fontSize: 12, fill: 'var(--text-secondary)' }}
                  tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip content={<CustomTooltip />} />
                {MOODS.map((m) => (
                  <Scatter
                    key={m}
                    name={m}
                    data={byMood[m]}
                    fill={MOOD_COLORS[m]}
                    opacity={0.7}
                    r={3}
                    onClick={(d) => setSelected(d)}
                    style={{ cursor: 'pointer' }}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {selected && (
            <div className={styles.sidePanel}>
              <button className={styles.closeBtn} onClick={() => setSelected(null)}>
                Close
              </button>
              <p className={styles.panelName}>{selected.track_name}</p>
              <p className={styles.panelArtist}>{selected.artists}</p>
              <p className={styles.panelMood}>{selected.mood}</p>
              <div className={styles.featureList}>
                {[
                  ['Energy', selected.energy],
                  ['Valence', selected.valence],
                  ['Danceability', selected.danceability],
                  ['Acousticness', selected.acousticness],
                  ['Instrumentalness', selected.instrumentalness],
                  ['Tempo', selected.tempo_norm],
                ].map(([label, val]) => (
                  <div key={label} className={styles.featureRow}>
                    <span className={styles.featureLabel}>{label}</span>
                    <div className={styles.featureBarTrack}>
                      <div
                        className={styles.featureBar}
                        style={{ width: `${Math.min(val * 100, 100)}%` }}
                      />
                    </div>
                    <span className={styles.featureVal}>{val.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </main>
  );
}
