import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { TrendingUp, Zap, Loader2, Sparkles } from 'lucide-react';

import { predictPropertyPrice, PredictionResult } from '../services/apiService';

export default function Predict() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  
  const [formData, setFormData] = useState({
    area: '',
    location: '',
    bedrooms: '3',
    bathrooms: '2',
    frontage: 'false',
    accessRoad: '10',
    floors: '2',
    legalStatus: 'Sale contract',
    furnitureState: 'Fully furnished',
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.area || !formData.location) return;

    const toOptionalNumber = (value: string): number | undefined => {
      const trimmed = value.trim();
      if (!trimmed) return undefined;

      const parsed = Number(trimmed);
      return Number.isNaN(parsed) ? undefined : parsed;
    };

    setLoading(true);
    try {
      const prediction = await predictPropertyPrice({
        area: Number(formData.area),
        location: formData.location,
        bedrooms: Number(formData.bedrooms),
        bathrooms: Number(formData.bathrooms),
        frontage: formData.frontage === 'true' ? 1 : 0,
        accessRoad: Number(formData.accessRoad),
        floors: Number(formData.floors),
        legalStatus: formData.legalStatus,
        furnitureState: formData.furnitureState,
      });
      setResult(prediction);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      alert(`Prediction failed: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background py-24 px-8">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
          <div className="lg:col-span-5">
            <span className="text-xs font-bold text-slate-500 tracking-widest uppercase block mb-4">Precision Valuation</span>
            <h1 className="text-6xl font-extrabold font-headline leading-tight tracking-tight text-primary mb-8">
              The Architect of <br /><span className="text-secondary">Value Prediction.</span>
            </h1>
            <p className="text-lg text-slate-600 font-medium leading-relaxed mb-12 max-w-md">
              Our proprietary AI engine leverages granular market datasets to provide valuation insights with professional-grade accuracy.
            </p>
            <div className="prediction-pulse bg-white p-6 rounded-lg shadow-sm border-l-4 border-secondary mb-6">
              <div className="flex items-center space-x-3 mb-2">
                <Zap className="text-secondary w-5 h-5 fill-secondary" />
                <span className="text-sm font-bold tracking-widest uppercase">Live Pulse</span>
              </div>
              <p className="text-sm text-slate-600 italic">"Current market trends show a 4.2% appreciation in suburban mid-century modern properties."</p>
            </div>
            
            {/* Dataset Info */}
            <div className="bg-slate-50 p-6 rounded-lg border border-slate-200">
              <div className="text-xs font-bold text-slate-600 tracking-widest uppercase mb-3">Dataset Summary (1200 Properties)</div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-slate-500">Area Range:</span>
                  <p className="font-bold text-primary">31 - 279 m²</p>
                </div>
                <div>
                  <span className="text-slate-500">Price Range:</span>
                  <p className="font-bold text-primary">2,034 - 34,857 tr VND</p>
                </div>
                <div>
                  <span className="text-slate-500">Locations:</span>
                  <p className="font-bold text-primary">10 Cities</p>
                </div>
                <div>
                  <span className="text-slate-500">Avg Accuracy:</span>
                  <p className="font-bold text-secondary">49.6%</p>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-7">
            <div className="bg-white p-10 rounded-xl shadow-sm border border-slate-100">
              <h2 className="font-headline text-2xl font-bold mb-8 text-primary flex items-center gap-2">
                <Sparkles className="w-6 h-6 text-secondary" />
                Property Specification
              </h2>
              <form className="space-y-6" onSubmit={handlePredict}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <div className="space-y-2">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Area (m2)</label>
                    <input 
                      name="area"
                      value={formData.area}
                      onChange={handleInputChange}
                      className="w-full bg-slate-50 border-none rounded-lg p-4 focus:ring-2 focus:ring-primary/20 transition-all" 
                      placeholder="e.g. 120 (range: 31-279)" 
                      type="number" 
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Location (Address)</label>
                    <input 
                      name="location"
                      value={formData.location}
                      onChange={handleInputChange}
                      className="w-full bg-slate-50 border-none rounded-lg p-4 focus:ring-2 focus:ring-primary/20 transition-all" 
                      placeholder="e.g. Quan 7, TP.HCM" 
                      type="text" 
                      required
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Beds</label>
                      <input 
                        name="bedrooms"
                        value={formData.bedrooms}
                        onChange={handleInputChange}
                        className="w-full bg-slate-50 border-none rounded-lg p-4 focus:ring-2 focus:ring-primary/20 transition-all" 
                        placeholder="1-7"
                        type="number" 
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Baths</label>
                      <input 
                        name="bathrooms"
                        value={formData.bathrooms}
                        onChange={handleInputChange}
                        className="w-full bg-slate-50 border-none rounded-lg p-4 focus:ring-2 focus:ring-primary/20 transition-all" 
                        placeholder="1-5"
                        type="number" 
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Frontage (Optional)</label>
                    <select
                      name="frontage"
                      value={formData.frontage}
                      onChange={handleInputChange}
                      className="w-full bg-slate-50 border-none rounded-lg p-4 focus:ring-2 focus:ring-primary/20 transition-all appearance-none"
                    >
                      <option value="false">No Frontage</option>
                      <option value="true">Has Frontage</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Floors</label>
                    <input
                      name="floors"
                      value={formData.floors}
                      onChange={handleInputChange}
                      className="w-full bg-slate-50 border-none rounded-lg p-4 focus:ring-2 focus:ring-primary/20 transition-all"
                      placeholder="1-5"
                      type="number"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Legal Status</label>
                    <select
                      name="legalStatus"
                      value={formData.legalStatus}
                      onChange={handleInputChange}
                      className="w-full bg-slate-50 border-none rounded-lg p-4 focus:ring-2 focus:ring-primary/20 transition-all appearance-none"
                    >
                      <option>Sale contract</option>
                      <option>Pink book</option>
                      <option>Red book</option>
                      <option>N/A</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Furniture State</label>
                    <select
                      name="furnitureState"
                      value={formData.furnitureState}
                      onChange={handleInputChange}
                      className="w-full bg-slate-50 border-none rounded-lg p-4 focus:ring-2 focus:ring-primary/20 transition-all appearance-none"
                    >
                      <option>N/A</option>
                      <option>Fully furnished</option>
                      <option>Basic furnished</option>
                      <option>Unfurnished</option>
                    </select>
                  </div>
                </div>
                <button 
                  className="w-full py-5 bg-secondary text-white rounded-xl font-bold text-lg hover:scale-[0.98] transition-transform shadow-lg shadow-secondary/20 flex items-center justify-center gap-3 disabled:opacity-70" 
                  type="submit"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin" />
                      Analyzing Market Data...
                    </>
                  ) : (
                    <>
                      <Zap className="w-6 h-6 fill-white" />
                      Predict Price
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>
        </div>

        {/* Results Section */}
        <AnimatePresence>
          {result && (
            <motion.section 
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-24 bg-slate-100 p-12 rounded-2xl"
            >
              <div className="flex items-end justify-between mb-12">
                <div>
                  <span className="text-xs font-bold text-secondary tracking-widest uppercase block mb-2">Analysis Complete</span>
                  <h2 className="text-4xl font-extrabold font-headline text-primary tracking-tight">Market Valuation Insight</h2>
                </div>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-7 bg-white p-10 rounded-xl relative overflow-hidden flex flex-col justify-between shadow-sm">
                  <div>
                    <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Estimated Value</span>
                    <div className="text-7xl font-black font-headline text-primary mt-4 tracking-tighter">
                      {(result.estimatedValue * 1000000).toLocaleString('vi-VN')} VND
                    </div>
                  </div>
                  <div className="mt-12 flex items-center space-x-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-secondary/10">
                      <TrendingUp className="text-secondary w-6 h-6" />
                    </div>
                    <p className="text-sm font-medium text-slate-600">Predicted to increase by <span className="text-secondary font-bold">{result.trend}%</span> in the next 12 months.</p>
                  </div>
                </div>
                <div className="lg:col-span-5 bg-white p-10 rounded-xl flex flex-col justify-center shadow-sm">
                  <span className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-6">Model Confidence</span>
                  <div className="flex items-end space-x-2 mb-4">
                    <span className="text-5xl font-black font-headline text-secondary">{result.confidence}%</span>
                    <span className="text-sm font-bold text-slate-400 mb-2 uppercase">Accuracy</span>
                  </div>
                  <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${result.confidence}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className="h-full bg-secondary"
                    ></motion.div>
                  </div>
                </div>
              </div>
              <div className="mt-8 bg-white p-10 rounded-xl shadow-sm">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Architectural & Market Analysis</h3>
                <p className="text-slate-600 leading-relaxed italic">
                  "{result.analysis}"
                </p>
              </div>
            </motion.section>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
