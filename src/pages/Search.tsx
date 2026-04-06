import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'motion/react';
import { Loader2, MapPin, Grid, Map as MapIcon, Bed, Bath, Square, Heart } from 'lucide-react';

import { fetchProperties, PropertyCard } from '../services/apiService';

function formatPriceInMillionVnd(price: number): string {
  return `${(price * 1000000).toLocaleString('vi-VN')} VNĐ`;
}

export default function SearchPage() {
  const [location, setLocation] = useState('');
  const [maxPrice, setMaxPrice] = useState(5000);
  const [properties, setProperties] = useState<PropertyCard[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadProperties = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetchProperties({
        location: location.trim() || undefined,
        maxPrice,
        limit: 24,
      });
      setProperties(response.items);
      setTotal(response.total);
    } catch (requestError) {
      setError('Cannot load property list. Check backend API connection.');
      setProperties([]);
      setTotal(0);
      console.error(requestError);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadProperties();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-8 py-12">
      <div className="flex flex-col md:flex-row gap-12">
        {/* Filters Sidebar */}
        <aside className="w-full md:w-80 flex-shrink-0">
          <div className="sticky top-24 space-y-8">
            <h2 className="text-2xl font-extrabold tracking-tight mb-6 font-headline text-primary">Refine Search</h2>
            
            <div className="space-y-3">
              <label className="text-xs uppercase tracking-widest font-bold text-slate-500">Location</label>
              <div className="relative">
                <input
                  className="w-full bg-white border border-slate-200 px-4 py-3 rounded-lg text-sm focus:ring-2 focus:ring-primary/20 outline-none"
                  placeholder="Enter city or district..."
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                />
                <MapPin className="absolute right-3 top-3 text-slate-400 w-5 h-5" />
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <label className="text-xs uppercase tracking-widest font-bold text-slate-500">Price Range</label>
                <span className="text-xs font-bold text-primary">Up to {formatPriceInMillionVnd(maxPrice)}</span>
              </div>
              <input
                className="w-full h-1.5 bg-slate-200 rounded-full appearance-none cursor-pointer accent-primary"
                type="range"
                min={500}
                max={20000}
                step={100}
                value={maxPrice}
                onChange={(e) => setMaxPrice(Number(e.target.value))}
              />
            </div>

            <button
              className="w-full bg-secondary text-white py-4 rounded-xl font-bold uppercase tracking-widest text-xs transition-all hover:opacity-90 shadow-lg shadow-secondary/20 disabled:opacity-70"
              onClick={loadProperties}
              disabled={loading}
            >
              Apply Filters
            </button>
          </div>
        </aside>

        {/* Main Content */}
        <section className="flex-1">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-10 gap-4">
            <div>
              <h1 className="text-4xl font-extrabold tracking-tight text-primary">Property Listings</h1>
              <p className="text-slate-500 text-sm mt-1">Found {total} properties from Catalyst dataset (54,202 total across 261 locations)</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center bg-slate-100 rounded-lg p-1">
                <button className="p-2 bg-white rounded shadow-sm text-primary">
                  <Grid className="w-5 h-5" />
                </button>
                <button className="p-2 text-slate-400 hover:text-primary transition-colors">
                  <MapIcon className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>

          {error && (
            <div className="mb-8 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {error}
            </div>
          )}

          {loading && (
            <div className="mb-8 flex items-center gap-3 text-sm text-slate-500">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading properties...
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {!loading && properties.length === 0 && (
              <div className="col-span-full rounded-xl border border-slate-200 bg-white p-8 text-center text-slate-500">
                No properties found. Try a different location or increase price range.
              </div>
            )}

            {properties.map((property, i) => (
              <motion.div 
                key={property.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="group bg-white rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-all"
              >
                <Link to={`/property/${property.id}`}>
                  <div className="relative h-64 overflow-hidden">
                    <img src="/pic/house.jpg" alt={property.title} className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" referrerPolicy="no-referrer" />
                    <div className="absolute top-4 left-4 flex gap-2">
                      <span className="bg-primary/90 text-white text-[10px] font-bold uppercase tracking-widest px-3 py-1.5 rounded backdrop-blur-md">{property.tag}</span>
                    </div>
                    <button className="absolute top-4 right-4 bg-white/20 backdrop-blur-md text-white p-2 rounded-full hover:bg-white/40 transition-all" onClick={(e) => e.preventDefault()}>
                      <Heart className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="p-6 space-y-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="text-xl font-extrabold text-primary tracking-tight">{property.title}</h3>
                        <p className="text-sm text-slate-500">{property.location}</p>
                      </div>
                      <div className="text-right">
                        <span className="text-2xl font-black text-primary">{formatPriceInMillionVnd(property.price)}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-6 py-4 border-y border-slate-100">
                      <div className="flex items-center gap-2 text-slate-600">
                        <Bed className="w-4 h-4" />
                        <span className="text-sm font-bold">{property.beds} Beds</span>
                      </div>
                      <div className="flex items-center gap-2 text-slate-600">
                        <Bath className="w-4 h-4" />
                        <span className="text-sm font-bold">{property.baths} Baths</span>
                      </div>
                      <div className="flex items-center gap-2 text-slate-600">
                        <Square className="w-4 h-4" />
                        <span className="text-sm font-bold">{property.sqft.toLocaleString('vi-VN')} m2</span>
                      </div>
                    </div>
                    <div className="prediction-pulse bg-slate-50 p-4 rounded-lg flex justify-between items-center">
                      <div>
                        <p className="text-[10px] uppercase tracking-widest font-bold text-slate-400">AI Prediction</p>
                        <p className="text-sm font-bold text-secondary">High Investment Potential</p>
                      </div>
                      <button className="text-primary font-bold text-xs hover:underline">View Insights</button>
                    </div>
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
