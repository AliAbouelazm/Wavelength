import styles from '../styles/components.module.css';

export default function ExplainBar({ item }) {
  const positive = item.direction === 'increases';
  return (
    <div className={styles.explainBar}>
      <span
        className={styles.explainDot}
        style={{ background: positive ? 'var(--positive-feature)' : 'var(--negative-feature)' }}
      />
      <span className={styles.explainPhrase}>{item.phrase}</span>
    </div>
  );
}
