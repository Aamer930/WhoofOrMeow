import { useState, useRef, useEffect } from 'react'

export default function TrainTab() {
  const [epochs, setEpochs] = useState(30)
  const [quick, setQuick] = useState(false)
  const [training, setTraining] = useState(false)
  const [log, setLog] = useState('')
  const logRef = useRef(null)

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [log])

  async function startTraining() {
    setTraining(true)
    const label = quick ? `⚡ Quick mode — 3 epochs, ~2k images` : `${epochs} epochs, 25k images`
    setLog(`🚀 Starting training — ${label}\n${'─'.repeat(45)}\n`)

    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs, quick }),
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const text = line.slice(6)
            if (text === '[DONE]') { setTraining(false); return }
            setLog((prev) => prev + text + '\n')
          }
        }
      }
    } catch (e) {
      setLog((prev) => prev + `\n❌ Connection error: ${e.message}\n`)
    } finally {
      setTraining(false)
    }
  }

  return (
    <div className="train-tab">
      <div className="train-header">
        <h2 className="train-title">Train a New Model</h2>
        <p className="train-desc">
          Trains on <code>data/dog-vs-cat/</code> with augmentation, BatchNorm,
          Dropout, and early stopping. Best weights saved to{' '}
          <code>models/best_model.keras</code>.
        </p>
      </div>

      <div className="train-controls">
        <div className="slider-group">
          <label className="slider-label">
            Max Epochs
            <span className="slider-value">{epochs}</span>
          </label>
          <input
            type="range"
            min={5}
            max={50}
            step={5}
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
            disabled={training}
            className="epoch-slider"
          />
          <div className="slider-ticks">
            {[5, 10, 20, 30, 40, 50].map((v) => (
              <span key={v}>{v}</span>
            ))}
          </div>
        </div>

        <div className="train-info">
          <div className="info-title">What happens</div>
          <ul className="info-list">
            <li>📦 Loads 25k images</li>
            <li>🔀 Applies augmentation</li>
            <li>🧠 Trains CNN (3 conv layers)</li>
            <li>🛑 Stops early if no improvement</li>
            <li>💾 Saves best model</li>
          </ul>
        </div>
      </div>

      <div className="train-actions">
        <button
          className={`train-btn ${training ? 'running' : ''}`}
          onClick={startTraining}
          disabled={training}
        >
          {training ? '⏳ Training…' : '🚂 Start Training'}
        </button>

        <label className="quick-toggle">
          <input
            type="checkbox"
            checked={quick}
            onChange={(e) => setQuick(e.target.checked)}
            disabled={training}
          />
          <span>
            ⚡ Quick mode
            <small> (~2 min, smoke test only)</small>
          </span>
        </label>
      </div>

      {log && (
        <div className="log-container" ref={logRef}>
          <pre className="log-text">{log}</pre>
        </div>
      )}
    </div>
  )
}
