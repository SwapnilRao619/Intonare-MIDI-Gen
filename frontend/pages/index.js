import { useState } from 'react';
import axios from 'axios';
import Particles from 'react-tsparticles';
import { loadFull } from 'tsparticles';
import styles from '../styles/Home.module.css';

export default function Home() {
  const [instrument, setInstrument] = useState('Piano');
  const [dynamics, setDynamics] = useState('mf');
  const [articulation, setArticulation] = useState('Staccato');
  const [loading, setLoading] = useState(false);
  const [downloadLink, setDownloadLink] = useState('');

  // Function to generate music by communicating with the backend
  const generateMusic = async () => {
    setLoading(true);
    setDownloadLink('');
    try {
      const response = await axios.post('http://localhost:5000/generate', {
        instrument,
        dynamics,
        articulation
      }, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      setDownloadLink(url);
    } catch (error) {
      console.error('Error generating music:', error);
    }
    setLoading(false);
  };

  // Initialize the tsParticles instance with all features
  const particlesInit = async (main) => {
    await loadFull(main);
  };

  return (
    <div className={styles.container}>
      {/* Particle Background */}
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={{
          background: {
            color: "#1a1a1a", // Darker background for better contrast
          },
          fpsLimit: 60,
          interactivity: {
            events: {
              onClick: { enable: true, mode: "push" },
              onHover: { enable: true, mode: "repulse" },
              resize: true,
            },
            modes: {
              push: { quantity: 4 },
              repulse: { distance: 200, duration: 0.4 },
            },
          },
          particles: {
            color: { value: "#ff4757" }, // Vibrant color for particles
            links: { color: "#ff6b81", distance: 150, enable: true, opacity: 0.5, width: 1 },
            collisions: { enable: true },
            move: { direction: "none", enable: true, outModes: "bounce", random: false, speed: 2, straight: false },
            number: { density: { enable: true, area: 800 }, value: 80 },
            opacity: { value: 0.7 },
            shape: { type: "circle" },
            size: { random: true, value: 5 },
          },
          detectRetina: true,
        }}
      />

      {/* Title */}
      <h1 className={styles.title}>Intonare [MIDI-Gen]</h1>

      {/* Selection Form */}
      <div className={styles.formGroup}>
        {/* Instrument Selection */}
        <label className={styles.label}>Instrument</label>
        <select value={instrument} onChange={(e) => setInstrument(e.target.value)} className={styles.select}>
          <option value="Piano">Piano</option>
          <option value="Violin">Violin</option>
          <option value="Flute">Flute</option>
          <option value="Guitar">Guitar</option>
        </select>
      </div>

      <div className={styles.formGroup}>
        {/* Dynamics Selection */}
        <label className={styles.label}>Dynamics</label>
        <select value={dynamics} onChange={(e) => setDynamics(e.target.value)} className={styles.select}>
          <option value="p">Piano (p)</option>
          <option value="mf">Mezzo-Forte (mf)</option>
          <option value="f">Forte (f)</option>
        </select>
      </div>

      <div className={styles.formGroup}>
        {/* Articulation Selection */}
        <label className={styles.label}>Articulation</label>
        <select value={articulation} onChange={(e) => setArticulation(e.target.value)} className={styles.select}>
          <option value="Staccato">Staccato</option>
          <option value="Tenuto">Tenuto</option>
          <option value="Accent">Accent</option>
        </select>
      </div>

      {/* Generate Button */}
      <button onClick={generateMusic} disabled={loading} className={`${styles.button} ${loading ? styles.loading : ''}`}>
        {loading ? 'Generating...' : 'Generate'}
      </button>

      {/* Download Link */}
      {downloadLink && (
        <div className={styles.downloadLink}>
          <a href={downloadLink} download="generated_song.mid">Download MIDI</a>
        </div>
      )}
    </div>
  );
}