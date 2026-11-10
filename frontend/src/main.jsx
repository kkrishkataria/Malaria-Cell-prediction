import React, { useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import './overrides.css'

const sampleMetrics = { architecture: 'Custom CNN', image_size: '128 x 128', dataset: 'NIH Malaria Cell Images', validation_accuracy: 'Train to generate' }

function App() {
  const input = useRef(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)

  const chooseFile = (picked) => {
    if (!picked) return
    if (!['image/jpeg', 'image/jpg', 'image/png'].includes(picked.type)) { setMessage('Please choose a PNG or JPEG image.'); return }
    setFile(picked); setPreview(URL.createObjectURL(picked)); setResult(null); setMessage('')
  }
  const analyze = async () => {
    if (!file) { input.current.click(); return }
    setLoading(true); setMessage('')
    const form = new FormData(); form.append('file', file)
    try {
      const response = await fetch('/api/predict', { method: 'POST', body: form })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail)
      setResult(data)
    } catch (error) { setMessage(error.message || 'Could not reach the local API.') }
    finally { setLoading(false) }
  }
  return <main>
    <nav><a className="brand" href="#top"><span className="mark">✦</span> CellScope <b>AI</b></a><span className="status"><i/> Clinical image classifier</span></nav>
    <section className="hero" id="top"><div className="eyebrow">TENSORFLOW · KERAS · CNN</div><h1>Clear insight from<br/><em>every cell.</em></h1><p>A focused image-classification workspace for screening malaria cell images with a custom convolutional neural network.</p><div className="hero-actions"><button onClick={analyze}>{loading ? 'Analyzing…' : 'Analyze a cell image'} <span>→</span></button></div><div className="orb orb-one"/><div className="orb orb-two"/></section>
    <section className="workspace"><div className="upload-card"><div className="card-head"><div><span className="section-tag">01 / IMAGE INPUT</span><h2>Screen a sample</h2></div><span className="format">PNG · JPG</span></div><input ref={input} type="file" accept="image/png,image/jpeg" onChange={e => chooseFile(e.target.files[0])}/><div className={'dropzone ' + (preview ? 'has-image' : '')} onClick={() => input.current.click()} onDragOver={e => e.preventDefault()} onDrop={e => {e.preventDefault(); chooseFile(e.dataTransfer.files[0])}}>{preview ? <img src={preview} alt="Selected cell sample"/> : <><div className="upload-icon">↑</div><strong>Drop a microscopic image here</strong><small>or click to browse your files</small></>}</div>{file && <div className="file-row"><span>✓</span><div><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toFixed(2)} MB · ready to analyze</small></div><button className="text-button" onClick={() => {setFile(null);setPreview(null);setResult(null)}}>Remove</button></div>}<button className="analyze" onClick={analyze} disabled={loading}>{loading ? 'Running CNN inference…' : 'Run analysis'} <span>→</span></button>{message && <p className="message">{message}</p>}</div>
    <div className="result-card"><span className="section-tag">02 / MODEL OUTPUT</span>{result ? <><div className={'result-icon ' + (result.prediction === 'Parasitized' ? 'alert' : 'safe')}>{result.prediction === 'Parasitized' ? '!' : '✓'}</div><h2>{result.prediction}</h2><p className="result-copy">{result.prediction === 'Parasitized' ? 'The model detected features associated with malaria infection.' : 'No malaria-associated features were detected by the model.'}</p><div className="confidence"><div><span>Model confidence</span><strong>{result.confidence}%</strong></div><div className="bar"><i style={{width: `${result.confidence}%`}}/></div></div><div className="risk"><span>Risk level</span><b>{result.risk_level}</b></div></> : <><div className="result-icon idle">⌁</div><h2>Awaiting sample</h2><p className="result-copy">Upload a cell image to receive a classification and confidence score.</p><div className="empty-lines"><i/><i/><i/></div></>}</div></section>
    <section className="model-strip" id="workflow"><div><span className="section-tag">MODEL PROFILE</span><h2>Built for a practical ML workflow.</h2></div><div className="metrics"><div><b>{sampleMetrics.architecture}</b><span>Architecture</span></div><div><b>{sampleMetrics.image_size}</b><span>Input resolution</span></div><div><b>{sampleMetrics.dataset}</b><span>Training data</span></div></div></section>
    <footer><span>CELLSCOPE AI / MALARIA CELL CLASSIFIER</span><span>For educational screening support only — not a clinical diagnosis.</span></footer>
  </main>
}
createRoot(document.getElementById('root')).render(<App />)
