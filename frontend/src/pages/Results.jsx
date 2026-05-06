import { useState } from 'react';
import { Link, Navigate, useLocation } from 'react-router-dom';
import TrackCard from '../components/TrackCard';
import styles from '../styles/results.module.css';

export default function Results() {
  const { state } = useLocation();
  const [sortBy, setSortBy] = useState('match');

  if (!state?.tracks) {
    return <Navigate to="/" replace />;
  }

  const { mood, tracks } = state;

  const sorted = [...tracks].sort((a, b) => {
    if (sortBy === 'match') return b.similarity_score - a.similarity_score;
    return b.energy - a.energy;
  });

  return (
    <main className={styles.page}>
      <Link to="/" className={styles.backLink}>Back</Link>

      <div className={styles.topRow}>
        <h1 className={styles.heading}>{mood}</h1>
        <div className={styles.sortControls}>
          <button
            className={sortBy === 'match' ? styles.sortActive : styles.sortBtn}
            onClick={() => setSortBy('match')}
          >
            Sort by match score
          </button>
          <button
            className={sortBy === 'energy' ? styles.sortActive : styles.sortBtn}
            onClick={() => setSortBy('energy')}
          >
            Sort by energy
          </button>
        </div>
      </div>

      <div className={styles.grid}>
        {sorted.map((track, i) => (
          <TrackCard key={`${track.track_name}-${i}`} track={track} />
        ))}
      </div>
    </main>
  );
}
