import React, { useState, useRef } from 'react';
import { Upload, Activity, ShieldCheck, Microscope, AlertCircle, RefreshCcw, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const App = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      setPreview(URL.createObjectURL(droppedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (err) {
      // Check if it's a network error (like backend not running)
      if (!err.response) {
        setError('Network error: Could not connect to the AI Backend. Is the python server running?');
      } else {
        setError(err.response?.data?.detail || 'An error occurred during prediction.');
      }
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans p-6 md:p-12 selection:bg-blue-500/30">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <header className="mb-12 text-center">
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-100 border border-blue-200 text-blue-700 mb-4 shadow-sm"
          >
            <Microscope size={18} />
            <span className="text-sm font-bold tracking-wide uppercase">Diagnostic Assistant</span>
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-4xl md:text-6xl font-extrabold mb-4 bg-gradient-to-r from-blue-700 via-sky-600 to-teal-500 bg-clip-text text-transparent"
          >
            Smart Malaria Detection
          </motion.h1>
          <p className="text-slate-500 text-lg max-w-2xl mx-auto font-medium">
            Quick and accurate cellular analysis to assist with malaria detection. Please upload a blood smear image for an instant assessment.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-5">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white border border-slate-200 rounded-3xl p-8 sticky top-8 shadow-xl"
            >
              <h2 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
                <Upload size={20} className="text-blue-600" />
                Sample Upload
              </h2>

              <div 
                onDragOver={onDragOver}
                onDrop={onDrop}
                onClick={() => fileInputRef.current.click()}
                className={`relative group border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 ${
                  preview ? 'border-blue-300 bg-blue-50/50' : 'border-slate-300 hover:border-blue-400 hover:bg-blue-50'
                }`}
              >
                <input 
                  type="file" 
                  className="hidden" 
                  ref={fileInputRef} 
                  onChange={handleFileChange}
                  accept="image/*"
                />
                
                {preview ? (
                  <div className="space-y-4">
                    <img src={preview} alt="Preview" className="w-full h-48 object-cover rounded-xl shadow-md border border-slate-200" />
                    <p className="text-sm text-slate-500 truncate font-medium">{file?.name}</p>
                  </div>
                ) : (
                  <div className="py-8">
                    <div className="w-16 h-16 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 group-hover:bg-blue-100 transition-all duration-300 shadow-sm">
                      <Upload size={28} />
                    </div>
                    <p className="text-slate-700 font-bold mb-1">Drag & Drop Blood Smear</p>
                    <p className="text-slate-500 text-sm">PNG, JPG up to 10MB</p>
                  </div>
                )}
              </div>

              <div className="mt-8 space-y-3">
                <button
                  onClick={handleUpload}
                  disabled={!file || loading}
                  className="w-full py-4 bg-gradient-to-r from-blue-600 to-sky-500 hover:from-blue-700 hover:to-sky-600 disabled:opacity-60 disabled:cursor-not-allowed rounded-xl font-bold text-white shadow-lg shadow-blue-500/30 transition-all duration-200 flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <RefreshCcw className="animate-spin" />
                  ) : (
                    <>
                      <Activity size={20} />
                      Analyze Specimen
                    </>
                  )}
                </button>
                
                {file && (
                  <button 
                    onClick={reset}
                    className="w-full py-3 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-xl text-sm font-bold transition-colors"
                  >
                    Clear Selection
                  </button>
                )}
              </div>
            </motion.div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-7">
            <AnimatePresence mode="wait">
              {loading ? (
                <motion.div 
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full min-h-[400px] flex flex-col items-center justify-center space-y-6 bg-white rounded-3xl border border-slate-200 shadow-lg"
                >
                  <div className="relative">
                    <div className="w-20 h-20 border-4 border-blue-500/20 rounded-full animate-ping absolute" />
                    <div className="w-20 h-20 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
                  </div>
                  <div className="text-center">
                    <p className="text-xl font-bold text-blue-700 mb-2">Analyzing Morphology...</p>
                    <p className="text-slate-500 font-medium">Extracting features and identifying pathogens</p>
                  </div>
                </motion.div>
              ) : error ? (
                <motion.div 
                  key="error"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-red-50 border border-red-200 rounded-3xl p-8 text-center shadow-lg"
                >
                  <AlertCircle size={56} className="text-red-500 mx-auto mb-4" />
                  <h3 className="text-2xl font-bold text-red-700 mb-3">Analysis Failed</h3>
                  <p className="text-red-600/80 mb-6 font-medium bg-red-100/50 p-4 rounded-xl">{error}</p>
                  <button onClick={reset} className="px-8 py-3 bg-red-600 hover:bg-red-700 text-white rounded-xl font-bold shadow-md shadow-red-500/20 transition-colors">Try Again</button>
                </motion.div>
              ) : result ? (
                <motion.div 
                  key="result"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-6"
                >
                  {/* Verdict Card */}
                  <div className="bg-white border border-slate-200 rounded-3xl p-8 shadow-xl">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
                      <div>
                        <p className="text-slate-500 text-sm uppercase tracking-widest font-bold mb-2">Diagnosis Verdict</p>
                        <h3 className={`text-4xl font-extrabold flex items-center gap-3 ${result.is_infected ? 'text-rose-600' : 'text-emerald-600'}`}>
                          {result.is_infected ? <AlertCircle size={36} /> : <CheckCircle2 size={36} />}
                          {result.prediction}
                        </h3>
                      </div>
                      <div className="bg-slate-50 rounded-2xl p-5 border border-slate-200 shadow-inner">
                        <p className="text-slate-500 text-xs uppercase font-bold mb-1">Diagnostic Confidence</p>
                        <p className="text-3xl font-mono font-black text-blue-700">{result.confidence}</p>
                      </div>
                    </div>

                    {/* Image Comparison */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-3">
                        <p className="text-sm font-bold text-slate-700 flex items-center gap-2">
                          <Microscope size={18} className="text-blue-600" /> Original Scan
                        </p>
                        <div className="rounded-2xl overflow-hidden border border-slate-200 shadow-md">
                          <img src={result.original_image} alt="Original" className="w-full aspect-square object-cover" />
                        </div>
                      </div>
                      <div className="space-y-3">
                        <p className="text-sm font-bold text-slate-700 flex items-center gap-2">
                          <ShieldCheck size={18} className="text-emerald-600" /> Detection Heatmap
                        </p>
                        <div className="rounded-2xl overflow-hidden border border-slate-200 shadow-md relative">
                          <img src={result.heatmap_image} alt="Heatmap" className="w-full aspect-square object-cover" />
                          <div className="absolute bottom-4 left-4 right-4 bg-white/90 backdrop-blur-md px-4 py-3 rounded-xl text-xs font-semibold text-slate-700 shadow-lg border border-slate-200/50">
                            <span className="flex items-center gap-2 mb-1"><div className="w-2 h-2 rounded-full bg-red-500"></div> High Activation</span>
                            Red areas highlight the regions associated with parasite signatures.
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Recommendations */}
                  <div className={`border rounded-3xl p-6 shadow-sm ${result.is_infected ? 'bg-rose-50 border-rose-200' : 'bg-emerald-50 border-emerald-200'}`}>
                    <h4 className={`font-bold mb-2 flex items-center gap-2 ${result.is_infected ? 'text-rose-700' : 'text-emerald-700'}`}>
                      <Activity size={20} /> Assistant Recommendation
                    </h4>
                    <p className={`text-sm leading-relaxed font-medium ${result.is_infected ? 'text-rose-800' : 'text-emerald-800'}`}>
                      {result.is_infected 
                        ? "URGENT: Parasitic morphology detected with high confidence. Immediate clinical correlation and confirmatory microscopy are recommended. Begin standard malaria protocol based on local guidelines."
                        : "Specimen appears consistent with healthy cellular morphology. No significant parasitic signatures identified. Routine follow-up may be maintained unless clinical symptoms persist."}
                    </p>
                  </div>
                </motion.div>
              ) : (
                <div className="h-full min-h-[400px] flex flex-col items-center justify-center bg-slate-50 rounded-3xl border-2 border-dashed border-slate-300">
                  <div className="w-24 h-24 rounded-full bg-white shadow-sm flex items-center justify-center mb-6">
                    <Microscope size={48} className="text-blue-300" />
                  </div>
                  <h3 className="text-xl font-bold text-slate-700 mb-2">Ready for Analysis</h3>
                  <p className="text-slate-500 font-medium text-center max-w-sm">
                    Upload a cellular image on the left to see the diagnostic breakdown here.
                  </p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <footer className="mt-20 py-8 border-t border-slate-200 text-center">
          <p className="text-slate-500 text-sm font-medium flex items-center justify-center gap-2">
            Built with <span className="text-rose-500">♥</span> for Medical Science & Innovation 2026
          </p>
        </footer>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        body { font-family: 'Inter', sans-serif; }
      `}} />
    </div>
  );
};

export default App;
