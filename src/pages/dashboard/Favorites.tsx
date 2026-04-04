import React from 'react';
import { motion } from 'motion/react';
import { 
  Heart, 
  Bell, 
  Trash2, 
  PlusCircle, 
  TrendingUp, 
  Square, 
  Bed, 
  Bath, 
  ChevronRight,
  FileText,
  HelpCircle,
  ArrowRight,
  LayoutGrid
} from 'lucide-react';

export default function Favorites() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-12"
    >
      <header>
        <p className="font-bold text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-2">Curated Collection</p>
        <h2 className="font-headline text-4xl font-black tracking-tight text-primary">Saved Portfolios</h2>
        <p className="text-slate-400 font-bold text-sm mt-4 max-w-2xl leading-relaxed">Organize your prospective investments and market monitoring filters in one architectural view.</p>
      </header>

      <div className="grid grid-cols-12 gap-10">
        {/* Left Column */}
        <div className="col-span-12 lg:col-span-4 space-y-10">
          <section className="bg-slate-50/50 p-10 rounded-3xl border border-slate-100">
            <div className="flex items-center justify-between mb-10">
              <h3 className="font-headline text-xl font-bold text-primary">Saved Searches</h3>
              <span className="text-[8px] font-black text-slate-400 bg-white px-3 py-1.5 rounded-full uppercase tracking-widest shadow-sm">3 Presets</span>
            </div>

            <div className="space-y-6">
              {[
                { title: 'Waterfront 3M+', tags: ['Miami', '>$3.0M'], active: true },
                { title: 'Modern SF', tags: ['SF Bay', 'Modern'], active: false },
              ].map((item) => (
                <div key={item.title} className="bg-white p-6 rounded-3xl border border-slate-50 shadow-sm hover:border-primary/20 transition-all group">
                  <div className="flex justify-between items-start mb-4">
                    <h4 className="font-headline font-black text-primary">{item.title}</h4>
                    <div className="flex gap-3">
                      <button className={`text-slate-300 hover:text-primary transition-colors ${item.active ? 'text-secondary' : ''}`}>
                        <Bell className="w-4 h-4" />
                      </button>
                      <button className="text-slate-300 hover:text-red-500 transition-colors">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2 mb-6">
                    {item.tags.map(tag => (
                      <span key={tag} className="bg-slate-50 text-[8px] font-black px-3 py-1 rounded-full uppercase tracking-widest text-slate-400">{tag}</span>
                    ))}
                  </div>
                  <button className="w-full py-3 bg-primary text-white rounded-2xl text-[10px] font-black uppercase tracking-widest hover:opacity-90 transition-all">
                    Execute Search
                  </button>
                </div>
              ))}
            </div>

            <button className="w-full mt-8 py-5 border-2 border-dashed border-slate-200 rounded-3xl flex items-center justify-center gap-3 text-slate-400 hover:border-primary hover:text-primary transition-all group">
              <PlusCircle className="w-5 h-5 group-hover:scale-110 transition-transform" />
              <span className="text-[10px] font-black uppercase tracking-widest">New Filter Preset</span>
            </button>
          </section>

          <section className="bg-primary text-white p-10 rounded-3xl flex flex-col relative overflow-hidden">
            <div className="relative z-10">
              <h3 className="font-headline text-xl font-bold mb-4">Architectural Support</h3>
              <p className="text-xs text-white/60 mb-8 leading-relaxed">Need help navigating deep market insights or custom API access for your portfolio?</p>
              <div className="space-y-4">
                <button className="flex items-center gap-3 text-[10px] font-black uppercase tracking-widest hover:text-secondary transition-all group">
                  <FileText className="w-4 h-4" /> Knowledge Base <ArrowRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
                </button>
                <button className="flex items-center gap-3 text-[10px] font-black uppercase tracking-widest hover:text-secondary transition-all group">
                  <HelpCircle className="w-4 h-4" /> Contact Specialist <ArrowRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
                </button>
              </div>
            </div>
            <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2"></div>
          </section>
        </div>

        {/* Right Column */}
        <div className="col-span-12 lg:col-span-8 space-y-12">
          <section>
            <div className="flex justify-between items-end mb-10">
              <div>
                <h3 className="font-headline text-2xl font-black text-primary">Saved Properties</h3>
                <p className="text-sm font-bold text-slate-400">Active monitoring for your architectural favorites</p>
              </div>
              <div className="flex gap-6">
                <button className="flex items-center gap-3 text-[10px] font-black text-primary uppercase tracking-widest hover:underline">
                  <TrendingUp className="w-4 h-4" /> Sort by Prediction
                </button>
                <button className="text-slate-300 hover:text-primary transition-all">
                  <LayoutGrid className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
              {[
                { title: 'Oak Ridge Estate', price: '$4,250,000', mls: '882910', location: 'Bel Air, Los Angeles, CA', growth: '+14%', beds: 5, baths: 6.5, area: '8.4k', image: '/pic/house.jpg' },
                { title: 'Glass Pavilion House', price: '$2,890,000', mls: '112039', location: 'Pacific Heights, San Francisco, CA', growth: 'High ROI', beds: 3, baths: 4, area: '3.2k', image: '/pic/house.jpg' },
                { title: 'The Concrete Loft', price: '$1,550,000', mls: '992011', location: 'East Austin, TX', growth: 'Target Met', beds: 2, baths: 2.5, area: '1.8k', image: '/pic/house.jpg' },
              ].map((item, idx) => (
                <div key={idx} className={`group bg-white rounded-3xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-500 border border-slate-50 ${idx === 2 ? 'border-l-4 border-secondary' : ''}`}>
                  <div className="h-64 relative overflow-hidden">
                    <img 
                      alt={item.title} 
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" 
                      src={item.image} 
                    />
                    <button className="absolute top-6 right-6 p-3 bg-white/90 backdrop-blur rounded-full text-red-500 shadow-lg">
                      <Heart className="w-5 h-5 fill-red-500" />
                    </button>
                    <div className="absolute bottom-6 left-6">
                      <span className={`${idx === 1 ? 'bg-primary' : 'bg-secondary'} text-white text-[8px] font-black px-4 py-1.5 rounded-full uppercase tracking-widest shadow-lg`}>
                        {item.growth} {idx === 0 ? 'Predicted Growth' : idx === 1 ? 'ROI Confidence' : 'Predictive Target Met'}
                      </span>
                    </div>
                  </div>
                  <div className="p-8">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-headline font-black text-2xl text-primary">{item.price}</h4>
                      <span className="text-[8px] font-black text-slate-300 uppercase tracking-widest mt-2">MLS: {item.mls}</span>
                    </div>
                    <p className="font-headline font-black text-lg text-primary mb-1">{item.title}</p>
                    <p className="text-sm font-bold text-slate-400 mb-8">{item.location}</p>
                    <div className="flex justify-between items-center pt-6 border-t border-slate-50">
                      <div className="flex gap-6 text-[10px] font-black text-slate-300 uppercase tracking-widest">
                        <span className="flex items-center gap-2"><Bed className="w-4 h-4" /> {item.beds}</span>
                        <span className="flex items-center gap-2"><Bath className="w-4 h-4" /> {item.baths}</span>
                        <span className="flex items-center gap-2"><Square className="w-4 h-4" /> {item.area}</span>
                      </div>
                      <button className="text-[10px] font-black text-primary uppercase tracking-widest hover:underline">Details</button>
                    </div>
                  </div>
                </div>
              ))}

              {/* Empty Slot */}
              <div className="bg-slate-50/50 rounded-3xl border-2 border-dashed border-slate-200 flex flex-col items-center justify-center p-12 text-center group hover:border-primary transition-all">
                <div className="w-20 h-20 rounded-full bg-white flex items-center justify-center mb-6 shadow-sm group-hover:scale-110 transition-transform">
                  <PlusCircle className="w-10 h-10 text-slate-200 group-hover:text-primary" />
                </div>
                <h5 className="font-headline font-black text-primary mb-2">Expand Your Portfolio</h5>
                <p className="text-xs font-bold text-slate-400 max-w-[200px] mb-8 leading-relaxed">Continue exploring high-growth architectural listings to add them here.</p>
                <button className="px-10 py-4 bg-primary text-white rounded-2xl text-[10px] font-black uppercase tracking-widest hover:shadow-xl hover:shadow-primary/20 transition-all">Go to Marketplace</button>
              </div>
            </div>
          </section>
        </div>
      </div>
    </motion.div>
  );
}
