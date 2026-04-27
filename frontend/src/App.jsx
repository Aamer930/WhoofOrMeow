import { useState } from 'react'
import PredictTab from './components/PredictTab'
import TrainTab from './components/TrainTab'

export default function App() {
  const [tab, setTab] = useState('predict')

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">
          Whoof<span className="title-accent">Or</span>Meow
          <span className="title-paw"> 🐾</span>
        </h1>
        <p className="subtitle">Drop a photo. We'll settle the debate.</p>

        <nav className="tabs">
          <button
            className={`tab-btn ${tab === 'predict' ? 'active' : ''}`}
            onClick={() => setTab('predict')}
          >
            🔍 Predict
          </button>
          <button
            className={`tab-btn ${tab === 'train' ? 'active' : ''}`}
            onClick={() => setTab('train')}
          >
            ⚙️ Train
          </button>
        </nav>
      </header>

      <main className="main">
        {tab === 'predict' ? <PredictTab /> : <TrainTab />}
      </main>
    </div>
  )
}
