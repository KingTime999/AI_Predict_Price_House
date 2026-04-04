import React, { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { Search, MapPin, TrendingUp, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

import { fetchFeaturedProperties, PropertyCard } from '../services/apiService';

function formatPriceInMillionVnd(price: number): string {
  return `${price.toLocaleString('vi-VN')} tr VND`;
}

export default function Home() {
  const [featuredProperties, setFeaturedProperties] = useState<PropertyCard[]>([]);
  const [currentCityImage, setCurrentCityImage] = useState<'tphcm' | 'hanoi'>('tphcm');

  useEffect(() => {
    const loadFeatured = async () => {
      try {
        const items = await fetchFeaturedProperties(3);
        setFeaturedProperties(items);
      } catch (error) {
        console.error(error);
      }
    };

    void loadFeatured();
  }, []);

  // Alternate between TPHCM and Hanoi images every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentCityImage((prev) => (prev === 'tphcm' ? 'hanoi' : 'tphcm'));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <header className="relative w-full h-[80vh] min-h-[600px] flex items-center overflow-hidden">
        <motion.img
          key={currentCityImage}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
          src={currentCityImage === 'tphcm' ? '/pic/tphcm.webp' : '/pic/hanoi.jpg'}
          alt={currentCityImage === 'tphcm' ? 'Ho Chi Minh City' : 'Hanoi'}
          className="absolute inset-0 w-full h-full object-cover"
          referrerPolicy="no-referrer"
        />
        <div className="absolute inset-0 bg-primary/40 mix-blend-multiply"></div>
        <div className="relative max-w-7xl mx-auto px-8 w-full">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-3xl"
          >
            <h1 className="font-headline text-5xl md:text-7xl font-extrabold tracking-tight text-white mb-6 leading-[1.1]">
              The Science of <span className="text-secondary-container">Luxury Living</span>.
            </h1>
            <p className="text-lg text-white/90 font-medium max-w-xl mb-12">
              Engineered precision for the discerning investor. Access proprietary market data and architectural excellence.
            </p>

            {/* Search Overlay */}
            <div className="glass-panel p-2 rounded-xl flex flex-col md:flex-row items-stretch gap-2 shadow-2xl">
              <div className="flex-1 flex items-center bg-white px-4 py-3 rounded-lg">
                <MapPin className="text-slate-400 mr-3 w-5 h-5" />
                <input className="bg-transparent border-none focus:ring-0 w-full text-slate-900 font-medium placeholder:text-slate-400" placeholder="Location or Neighborhood" type="text" />
              </div>
              <div className="w-full md:w-48 flex items-center bg-white px-4 py-3 rounded-lg">
                <Search className="text-slate-400 mr-3 w-5 h-5" />
                <select className="bg-transparent border-none focus:ring-0 w-full text-slate-900 font-medium">
                  <option>Property Type</option>
                  <option>Villa</option>
                  <option>Penthouse</option>
                </select>
              </div>
              <button className="bg-primary text-white px-10 py-4 rounded-lg font-bold flex items-center justify-center transition-transform active:scale-95">
                Discover
              </button>
            </div>
          </motion.div>
        </div>
      </header>

      {/* Methodology Section */}
      <section className="py-24 max-w-7xl mx-auto px-8">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-12 items-center">
          <div className="md:col-span-5">
            <div className="inline-block py-1 px-3 bg-secondary-container text-secondary text-xs font-bold tracking-widest uppercase rounded mb-6">
              Our Methodology
            </div>
            <h2 className="font-headline text-4xl font-extrabold text-primary mb-8 tracking-tight">
              Beyond Simple Brokerage: <br />The Architectural Perspective.
            </h2>
            <div className="space-y-8">
              <div className="flex gap-6">
                <div className="w-12 h-12 bg-surface-container-high rounded-lg flex items-center justify-center shrink-0">
                  <TrendingUp className="text-primary w-6 h-6" />
                </div>
                <div>
                  <h4 className="font-headline font-bold text-lg text-primary mb-1">Data-Driven Predictions</h4>
                  <p className="text-slate-600 leading-relaxed">Our proprietary AI analyzes 50+ architectural variables to predict future market valuations with 94% accuracy.</p>
                </div>
              </div>
            </div>
          </div>
          <div className="md:col-span-7 relative">
            <div className="aspect-video rounded-xl overflow-hidden shadow-2xl">
              <img src="/pic/house.jpg" alt="Interior" className="w-full h-full object-cover" referrerPolicy="no-referrer" />
            </div>
            <div className="absolute -bottom-8 -left-8 glass-panel p-6 rounded-xl shadow-xl max-w-xs border-l-4 border-secondary">
              <div className="flex items-center gap-3 mb-2 text-secondary">
                <TrendingUp className="w-4 h-4" />
                <span className="text-xs font-bold tracking-widest uppercase">Live Insight</span>
              </div>
              <p className="font-headline font-bold text-primary mb-2">Valuation Surge Predicted</p>
              <p className="text-xs text-slate-600 leading-relaxed">Residential properties in the West Sector are projected to appreciate by 12.4% by Q4 2025.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Featured Collection */}
      <section className="bg-surface-container-low py-24">
        <div className="max-w-7xl mx-auto px-8">
          <div className="flex justify-between items-end mb-16">
            <div>
              <h2 className="font-headline text-4xl font-extrabold text-primary mb-2">Curated Collection</h2>
              <p className="text-slate-600 font-medium">Hand-selected for architectural merit and investment potential.</p>
            </div>
            <Link to="/search" className="text-primary font-bold flex items-center hover:gap-2 transition-all">
              View Portfolio <ArrowRight className="ml-1 w-5 h-5" />
            </Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {featuredProperties.map((prop) => (
              <Link key={prop.id} to={`/property/${prop.id}`} className="bg-white rounded-xl overflow-hidden group shadow-sm hover:shadow-md transition-all">
                <div className="relative h-72 overflow-hidden">
                  <img src={prop.image} alt={prop.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" referrerPolicy="no-referrer" />
                  <div className="absolute top-4 left-4 bg-primary text-white text-[10px] font-bold px-3 py-1 rounded-full">NEW LISTING</div>
                </div>
                <div className="p-6">
                  <h3 className="font-headline font-extrabold text-xl text-primary">{prop.title}</h3>
                  <p className="text-sm text-slate-500 mb-4">{prop.location}</p>
                  <div className="flex justify-between items-center pt-4 border-t border-slate-100">
                    <span className="text-xl font-black text-primary">{formatPriceInMillionVnd(prop.price)}</span>
                    <div className="flex gap-3 text-xs font-bold text-slate-400">
                      <span>{prop.beds} Bed</span>
                      <span>{prop.baths} Bath</span>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
