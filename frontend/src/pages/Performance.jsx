import { useEffect, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchModelMetrics } from '../api/client';
import styles from '../styles/performance.module.css';

const MODEL_LABELS = {
  random_forest: 'Random Forest',
  xgboost: 'XGBoost',
  mlp: 'MLP',
};

const BAR_COLORS = {
  random_forest: '#d97706',
  xgboost: '#60a5fa',
  mlp: '#a78bfa',
};

const CLASS_ORDER = ['Happy', 'Energetic', 'Melancholic', 'Focused', 'Calm', 'Intense', 'Neutral'];

function pct(val) {
  return (val * 100).toFixed(1) + '%';
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className={styles.tooltip}>
      <p className={styles.tooltipLabel}>{MODEL_LABELS[label] || label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color, margin: '2px 0', fontSize: 13 }}>
          {p.name}: {(p.value * 100).toFixed(2)}%
        </p>
      ))}
    </div>
  );
}

export default function Performance() {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelMetrics()
      .then(setMetrics)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className={styles.page}><p className={styles.error}>{error}</p></div>;
  if (!metrics) return <div className={styles.page}><p className={styles.loading}>Loading metrics...</p></div>;

  const { models, best_model } = metrics;

  const barData = Object.entries(models).map(([key, m]) => ({
    key,
    f1_macro: m.f1_macro,
    accuracy: m.accuracy,
  }));

  const bestReport = models[best_model]?.classification_report;
  const classRows = CLASS_ORDER.filter((c) => bestReport?.[c]);

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1 className={styles.heading}>Model Performance</h1>
        <p className={styles.subheading}>
          Comparison of Random Forest, XGBoost, and MLP trained on 89k Spotify tracks across 7 mood classes.
        </p>
      </div>

      <div className={styles.statRow}>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Best Model</span>
          <span className={styles.statValue}>{MODEL_LABELS[best_model] || best_model}</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>F1 Macro</span>
          <span className={styles.statValue}>{pct(models[best_model].f1_macro)}</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Accuracy</span>
          <span className={styles.statValue}>{pct(models[best_model].accuracy)}</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Test Samples</span>
          <span className={styles.statValue}>13,462</span>
        </div>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionHeading}>F1 Macro vs Accuracy by Model</h2>
        <div className={styles.chartWrap}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} barCategoryGap="30%" barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
              <XAxis
                dataKey="key"
                tickFormatter={(k) => MODEL_LABELS[k] || k}
                tick={{ fill: '#737373', fontSize: 13 }}
                axisLine={{ stroke: '#262626' }}
                tickLine={false}
              />
              <YAxis
                domain={[0.9, 1.0]}
                tickFormatter={(v) => (v * 100).toFixed(0) + '%'}
                tick={{ fill: '#737373', fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                width={40}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
              <Bar dataKey="f1_macro" name="F1 Macro" radius={[4, 4, 0, 0]}>
                {barData.map((entry) => (
                  <Cell key={entry.key} fill={BAR_COLORS[entry.key]} />
                ))}
              </Bar>
              <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]} fill="#404040">
                {barData.map((entry) => (
                  <Cell key={entry.key} fill={BAR_COLORS[entry.key] + '66'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className={styles.chartLegend}>
            <span className={styles.legendSolid}>Solid = F1 Macro</span>
            <span className={styles.legendFaded}>Faded = Accuracy</span>
          </div>
        </div>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionHeading}>Per-class F1 ({MODEL_LABELS[best_model]})</h2>
        <table className={styles.table}>
          <thead>
            <tr>
              <th className={styles.th}>Mood</th>
              <th className={styles.th}>Precision</th>
              <th className={styles.th}>Recall</th>
              <th className={styles.th}>F1</th>
              <th className={styles.th}>Support</th>
            </tr>
          </thead>
          <tbody>
            {classRows.map((cls) => {
              const r = bestReport[cls];
              return (
                <tr key={cls} className={styles.tr}>
                  <td className={styles.td}>{cls}</td>
                  <td className={styles.td}>{pct(r.precision)}</td>
                  <td className={styles.td}>{pct(r.recall)}</td>
                  <td className={`${styles.td} ${styles.f1Cell}`}>{pct(r['f1-score'])}</td>
                  <td className={styles.td}>{r.support.toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionHeading}>All Models</h2>
        <table className={styles.table}>
          <thead>
            <tr>
              <th className={styles.th}>Model</th>
              <th className={styles.th}>F1 Macro</th>
              <th className={styles.th}>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(models).map(([key, m]) => (
              <tr key={key} className={`${styles.tr} ${key === best_model ? styles.trBest : ''}`}>
                <td className={styles.td}>
                  {MODEL_LABELS[key] || key}
                  {key === best_model && <span className={styles.bestBadge}>active</span>}
                </td>
                <td className={styles.td}>{pct(m.f1_macro)}</td>
                <td className={styles.td}>{pct(m.accuracy)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
