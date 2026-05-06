import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import Nav from './components/Nav';
import Explore from './pages/Explore';
import Home from './pages/Home';
import Performance from './pages/Performance';
import Results from './pages/Results';

export default function App() {
  return (
    <BrowserRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/results" element={<Results />} />
        <Route path="/explore" element={<Explore />} />
        <Route path="/performance" element={<Performance />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
