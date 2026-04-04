import React from 'react';
import { motion } from 'motion/react';
import { 
  History, 
  TrendingUp, 
  BarChart3, 
  MapPin, 
  Square, 
  Bed, 
  CheckCircle2, 
  Filter,
  Eye,
  FileText,
  HelpCircle,
  ArrowRight
} from 'lucide-react';

export default function Activity() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-12"
    >
      <header>
        <p className="font-bold text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-2">Architectural Portal</p>
        <h2 className="font-headline text-4xl font-black tracking-tight text-primary">Activity History</h2>
        <p className="text-slate-400 font-bold text-sm mt-2">Review your architectural footprint and predictive intelligence.</p>
      </header>

      <div className="grid grid-cols-12 gap-10">
        {/* Left Column */}
        <div className="col-span-12 lg:col-span-8 space-y-12">
          {/* Recently Viewed */}
          <section>
            <div className="flex justify-between items-end mb-8">
              <h3 className="font-headline text-2xl font-black text-primary">Recently Viewed Properties</h3>
              <button className="text-xs font-black text-primary uppercase tracking-widest border-b-2 border-primary pb-1">View All History</button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="group bg-white rounded-3xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-500 border border-slate-50">
                <div className="h-56 relative overflow-hidden">
                  <img 
                    alt="Modern Villa" 
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" 
                    src="https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&q=80&w=800" 
                  />
                  <div className="absolute top-6 left-6 bg-white/90 backdrop-blur-md px-4 py-1.5 rounded-full text-[10px] font-black text-primary uppercase tracking-widest shadow-lg">
                    For Sale
                  </div>
                </div>
                <div className="p-8">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-headline font-black text-xl text-primary truncate">The Obsidian Heights</h4>
                    <p className="font-headline font-black text-primary">$2,450,000</p>
                  </div>
                  <p className="text-sm font-bold text-slate-400 mb-6 flex items-center gap-2">
                    <MapPin className="w-4 h-4" /> Beverly Hills, CA
                  </p>
                  <div className="flex gap-6 text-[10px] font-black text-slate-300 uppercase tracking-widest border-t border-slate-50 pt-6">
                    <span className="flex items-center gap-2"><Square className="w-4 h-4" /> 4,200 sqft</span>
                    <span className="flex items-center gap-2"><Bed className="w-4 h-4" /> 5 Beds</span>
                  </div>
                </div>
              </div>

              <div className="group bg-white rounded-3xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-500 border border-slate-50">
                <div className="h-56 relative overflow-hidden">
                  <img 
                    alt="Luxury Estate" 
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" 
                    src="https://images.unsplash.com/photo-1600566753376-12c8ab7fb75b?auto=format&fit=crop&q=80&w=800" 
                  />
                  <div className="absolute top-6 left-6 bg-white/90 backdrop-blur-md px-4 py-1.5 rounded-full text-[10px] font-black text-primary uppercase tracking-widest shadow-lg">
                    Pending
                  </div>
                </div>
                <div className="p-8">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-headline font-black text-xl text-primary truncate">Harbor Light Estate</h4>
                    <p className="font-headline font-black text-primary">$1,895,000</p>
                  </div>
                  <p className="text-sm font-bold text-slate-400 mb-6 flex items-center gap-2">
                    <MapPin className="w-4 h-4" /> Newport, RI
                  </p>
                  <div className="flex gap-6 text-[10px] font-black text-slate-300 uppercase tracking-widest border-t border-slate-50 pt-6">
                    <span className="flex items-center gap-2"><Square className="w-4 h-4" /> 3,150 sqft</span>
                    <span className="flex items-center gap-2"><Bed className="w-4 h-4" /> 4 Beds</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Price Prediction Analysis */}
          <section>
            <div className="flex justify-between items-end mb-8">
              <h3 className="font-headline text-2xl font-black text-primary">Price Prediction Analysis</h3>
              <button className="bg-slate-50 px-5 py-2.5 rounded-2xl text-[10px] font-black text-primary uppercase tracking-widest flex items-center gap-3 border border-slate-100 hover:bg-white transition-all">
                <Filter className="w-4 h-4" /> Filter Analysis
              </button>
            </div>
            
            <div className="space-y-4">
              {[
                { address: '842 Altamont Ridge, Beverly Hills', date: 'Oct 24, 2023', accuracy: '98.4%', actual: '$2.45M', predicted: '$2.41M' },
                { address: '12 Lighthouse Way, Newport, RI', date: 'Oct 18, 2023', accuracy: '96.1%', actual: '$1.89M', predicted: '$1.85M' },
              ].map((item, idx) => (
                <div key={idx} className="bg-slate-50/50 rounded-3xl border-l-4 border-secondary p-8 flex flex-col md:flex-row md:items-center justify-between gap-8 hover:bg-white hover:shadow-sm transition-all border border-slate-100">
                  <div className="flex items-center gap-6">
                    <div className="w-16 h-16 rounded-2xl bg-white flex items-center justify-center text-secondary shadow-sm">
                      <TrendingUp className="w-8 h-8" />
                    </div>
                    <div>
                      <h4 className="font-bold text-primary">{item.address}</h4>
                      <div className="flex items-center gap-4 mt-1">
                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Analyzed {item.date}</p>
                        <span className="px-2 py-0.5 bg-emerald-50 text-emerald-600 text-[8px] font-black rounded uppercase tracking-widest">Accuracy: {item.accuracy}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-12">
                    <div className="text-right">
                      <p className="text-[10px] font-black uppercase tracking-widest text-slate-300 mb-1">Actual List</p>
                      <p className="font-headline font-black text-xl text-primary">{item.actual}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] font-black uppercase tracking-widest text-secondary mb-1">Predicted</p>
                      <p className="font-headline font-black text-xl text-secondary">{item.predicted}</p>
                    </div>
                    <button className="p-3 bg-emerald-50 text-secondary rounded-full shadow-sm">
                      <CheckCircle2 className="w-6 h-6" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Right Column */}
        <div className="col-span-12 lg:col-span-4 space-y-10">
          {/* Activity Snapshot */}
          <section className="bg-white p-10 rounded-3xl shadow-[0_20px_50px_rgba(0,0,0,0.02)] border border-slate-50 border-l-4 border-primary">
            <h3 className="font-headline text-xl font-bold text-primary mb-10">Activity Snapshot</h3>
            <div className="space-y-8">
              <div className="flex justify-between items-center p-6 bg-slate-50/50 rounded-2xl border border-slate-100">
                <div>
                  <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Properties Viewed</p>
                  <p className="text-4xl font-black text-primary font-headline">42</p>
                </div>
                <Eye className="w-12 h-12 text-primary opacity-10" />
              </div>
              <div className="flex justify-between items-center p-6 bg-slate-50/50 rounded-2xl border border-slate-100">
                <div>
                  <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Total Predictions</p>
                  <p className="text-4xl font-black text-secondary font-headline">18</p>
                </div>
                <BarChart3 className="w-12 h-12 text-secondary opacity-20" />
              </div>
            </div>
          </section>

          {/* Architectural Support */}
          <section className="bg-primary text-white p-10 rounded-3xl flex flex-col relative overflow-hidden">
            <div className="relative z-10">
              <h3 className="font-headline text-xl font-bold mb-4">Architectural Support</h3>
              <p className="text-xs text-white/60 mb-8 leading-relaxed">Need help navigating deep market insights or custom API access for historical data?</p>
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
      </div>
    </motion.div>
  );
}
