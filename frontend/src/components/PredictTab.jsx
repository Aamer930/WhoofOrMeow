import { useState, useRef, useCallback } from 'react'

const VERDICTS = {
  dog_high: 'Woof! Definitely a dog.',
  dog_low: 'Probably a dog, but a stylish one.',
  cat_high: 'Purrfectly a cat.',
  cat_low: 'Leaning cat. Might just have dog energy.',
}

export default function PredictTab() {
  const [dragging, setDragging] = useState(false)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const processFile = useCallback(async (file) => {
    if (!file || !file.type.startsWith('image/')) return
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setLoading(true)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch('/api/predict', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Prediction failed')
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    processFile(e.dataTransfer.files[0])
  }, [processFile])

  const verdict = result
    ? result.is_dog
      ? result.confidence > 85 ? VERDICTS.dog_high : VERDICTS.dog_low
      : result.confidence > 85 ? VERDICTS.cat_high : VERDICTS.cat_low
    : null

  const accent = result
    ? result.is_dog ? 'var(--dog)' : 'var(--cat)'
    : 'var(--neutral)'

  return (
    <div className="predict-tab">
      {/* Drop zone */}
      <div
        className={`dropzone ${dragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => processFile(e.target.files[0])}
        />
        {preview ? (
          <img src={preview} alt="upload" className="preview-img" />
        ) : (
          <div className="dropzone-hint">
            <span className="dropzone-icon">🐾</span>
            <span className="dropzone-text">Drop a photo or click to browse</span>
            <span className="dropzone-sub">JPG, PNG, WEBP</span>
          </div>
        )}
        {loading && <div className="loading-overlay"><div className="spinner" /></div>}
      </div>

      {/* Error */}
      {error && (
        <div className="error-card">⚠️ {error}</div>
      )}

      {/* Results */}
      {result && (
        <div className="results">
          {/* Result card */}
          <div className="result-card" style={{ '--accent': accent }}>
            <div className="result-emoji">{result.is_dog ? '🐶' : '🐱'}</div>
            <div className="result-label" style={{ color: accent }}>
              {result.label.toUpperCase()}
            </div>
            <div className="confidence-row">
              <span className="confidence-text">Confidence</span>
              <span className="confidence-pct" style={{ color: accent }}>
                {result.confidence}%
              </span>
            </div>
            <div className="confidence-track">
              <div
                className="confidence-fill"
                style={{ width: `${result.confidence}%`, background: accent }}
              />
            </div>
            <div className="verdict">{verdict}</div>
          </div>

          {/* Grad-CAM */}
          <div className="gradcam-card">
            <div className="card-label">Grad-CAM — where the model looked</div>
            <img
              src={`data:image/png;base64,${result.gradcam}`}
              alt="Grad-CAM heatmap"
              className="gradcam-img"
            />
            <div className="gradcam-hint">Warm colours = high attention</div>
          </div>
        </div>
      )}
    </div>
  )
}
