import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart as RechartsRadar,
  ResponsiveContainer,
} from 'recharts';

export default function RadarChart({ track }) {
  const data = [
    { feature: 'Energy', value: track.energy },
    { feature: 'Valence', value: track.valence },
    { feature: 'Dance', value: track.danceability },
    { feature: 'Acoustic', value: track.acousticness },
    { feature: 'Instru', value: track.instrumentalness },
    { feature: 'Tempo', value: track.tempo_norm },
  ];

  return (
    <ResponsiveContainer width="100%" height={160}>
      <RechartsRadar outerRadius={52} data={data}>
        <PolarGrid stroke="var(--border)" />
        <PolarAngleAxis
          dataKey="feature"
          tick={{ fontSize: 9, fill: 'var(--text-secondary)' }}
        />
        <Radar
          dataKey="value"
          stroke="var(--accent)"
          fill="var(--accent)"
          fillOpacity={0.25}
          strokeWidth={1.5}
        />
      </RechartsRadar>
    </ResponsiveContainer>
  );
}
