import styles from '../styles/components.module.css';

const MOOD_DESCRIPTIONS = {
  Happy: 'Upbeat, bright, high energy with warm emotional tone',
  Energetic: 'Driving rhythms, peak intensity, built for movement',
  Melancholic: 'Quiet, introspective, low energy with emotional weight',
  Focused: 'Instrumental, mid-tempo, minimal distractions',
  Calm: 'Acoustic, gentle, low energy with peaceful texture',
  Intense: 'High energy, dark tone, emotionally charged',
};

export default function MoodCard({ mood, selected, onClick }) {
  const key = mood.toLowerCase();
  const cardStyle = {
    background: `var(--mood-${key}-bg)`,
    color: `var(--mood-${key}-text)`,
    borderColor: selected ? 'var(--accent)' : `color-mix(in srgb, var(--mood-${key}-text) 20%, transparent)`,
    borderBottomWidth: selected ? '2px' : '1px',
    borderBottomColor: selected ? 'var(--accent)' : undefined,
  };

  return (
    <button className={styles.moodCard} style={cardStyle} onClick={onClick}>
      <span className={styles.moodName}>{mood}</span>
      <span className={styles.moodDesc} style={{ opacity: 0.8 }}>
        {MOOD_DESCRIPTIONS[mood]}
      </span>
    </button>
  );
}
