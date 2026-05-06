import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import MoodGrid from '../components/MoodGrid';
import { recommend } from '../api/client';
import styles from '../styles/home.module.css';

export default function Home() {
  const navigate = useNavigate();
  const [selectedMood, setSelectedMood] = useState(null);
  const [energy, setEnergy] = useState(0.5);
  const [valence, setValence] = useState(0.5);
  const [tempo, setTempo] = useState(120);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleFindTracks() {
    if (!selectedMood) return;
    setLoading(true);
    setError(null);
    try {
      const data = await recommend({
        mood: selectedMood,
        energy_filter: energy,
        valence_filter: valence,
        tempo_filter: tempo,
      });
      navigate('/results', { state: { mood: selectedMood, tracks: data.tracks } });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className={styles.page}>
      <section className={styles.hero}>
        <h1 className={styles.heading}>Find music that fits how you want to feel.</h1>
        <p className={styles.subheading}>
          Pick a mood. Adjust the feel. Get tracks the model actually selected for you.
        </p>
      </section>

      <section className={styles.section}>
        <MoodGrid selected={selectedMood} onSelect={setSelectedMood} />
      </section>

      {selectedMood && (
        <section className={styles.section}>
          <div className={styles.sliders}>
            <div className={styles.sliderRow}>
              <label className={styles.sliderLabel}>Energy</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={energy}
                onChange={(e) => setEnergy(parseFloat(e.target.value))}
                className={styles.slider}
              />
              <span className={styles.sliderValue}>{energy.toFixed(2)}</span>
            </div>

            <div className={styles.sliderRow}>
              <label className={styles.sliderLabel}>Mood tone</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={valence}
                onChange={(e) => setValence(parseFloat(e.target.value))}
                className={styles.slider}
              />
              <span className={styles.sliderValue}>{valence.toFixed(2)}</span>
            </div>

            <div className={styles.sliderRow}>
              <label className={styles.sliderLabel}>Tempo</label>
              <input
                type="range"
                min="60"
                max="180"
                step="1"
                value={tempo}
                onChange={(e) => setTempo(parseInt(e.target.value, 10))}
                className={styles.slider}
              />
              <span className={styles.sliderValue}>{tempo} bpm</span>
            </div>
          </div>

          {error && <p className={styles.error}>{error}</p>}

          <button
            className={styles.findButton}
            onClick={handleFindTracks}
            disabled={loading}
          >
            {loading ? 'Finding tracks...' : 'Find Tracks'}
          </button>
        </section>
      )}
    </main>
  );
}
