import { NavLink } from 'react-router-dom';
import styles from '../styles/components.module.css';

export default function Nav() {
  const linkClass = ({ isActive }) =>
    isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink;

  return (
    <header className={styles.nav}>
      <span className={styles.wordmark}>Wavelength</span>
      <nav className={styles.navLinks}>
        <NavLink to="/" end className={linkClass}>
          Home
        </NavLink>
        <NavLink to="/explore" className={linkClass}>
          Explore
        </NavLink>
        <NavLink to="/performance" className={linkClass}>
          Performance
        </NavLink>
      </nav>
    </header>
  );
}
