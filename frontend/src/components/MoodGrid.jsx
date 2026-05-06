import MoodCard from './MoodCard';
import styles from '../styles/components.module.css';

const MOODS = ['Happy', 'Energetic', 'Melancholic', 'Focused', 'Calm', 'Intense'];

export default function MoodGrid({ selected, onSelect }) {
  return (
    <div className={styles.moodGrid}>
      {MOODS.map((mood) => (
        <MoodCard
          key={mood}
          mood={mood}
          selected={selected === mood}
          onClick={() => onSelect(mood)}
        />
      ))}
    </div>
  );
}
