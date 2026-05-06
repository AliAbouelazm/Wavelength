import ExplainBar from './ExplainBar';
import RadarChart from './RadarChart';
import styles from '../styles/components.module.css';

export default function TrackCard({ track }) {
  const matchPct = Math.round(track.similarity_score * 100);

  return (
    <div className={styles.trackCard}>
      <div className={styles.trackHeader}>
        <div className={styles.trackMeta}>
          <p className={styles.trackName}>{track.track_name}</p>
          <p className={styles.trackArtist}>{track.artists}</p>
        </div>
        <span className={styles.matchPill}>{matchPct}% match</span>
      </div>

      <RadarChart track={track} />

      <div className={styles.explainList}>
        {track.explanation.map((item, i) => (
          <ExplainBar key={i} item={item} />
        ))}
      </div>
    </div>
  );
}
