import React from 'react';
import { motion } from 'motion/react';
import { 
  Edit3, 
  Lock, 
  Bell, 
  CreditCard, 
  ShieldCheck, 
  ChevronRight,
  TrendingUp,
  BarChart3,
  FileText,
  Square,
  Bed,
  Bath,
  MapPin,
  Heart,
  X
} from 'lucide-react';

export default function Profile() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-12"
    >
      <header>
        <p className="font-bold text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-2">Architectural Portal</p>
        <h2 className="font-headline text-4xl font-black tracking-tight text-primary">Member Dashboard</h2>
      </header>

      <div className="grid grid-cols-12 gap-10">
        {/* Left Column */}
        <div className="col-span-12 lg:col-span-4 space-y-8">
          <section className="bg-white p-8 rounded-3xl shadow-[0_20px_50px_rgba(0,0,0,0.02)] border border-slate-50 border-l-4 border-primary">
            <div className="flex justify-between items-start mb-8">
              <h3 className="font-headline text-lg font-bold text-primary">Personal Profile</h3>
              <button className="text-slate-300 hover:text-primary transition-colors">
                <Edit3 className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-8">
              <div className="flex items-center gap-6">
                <div className="w-24 h-24 rounded-full overflow-hidden border-4 border-slate-50 shadow-inner">
                  <img 
                    alt="Alex Wright" 
                    className="w-full h-full object-cover" 
                    src="/static/house.jpg" 
                    referrerPolicy="no-referrer"
                  />
                </div>
                <div>
                  <p className="font-headline font-black text-2xl text-primary">Alex Wright</p>
                  <p className="text-sm font-bold text-slate-400">Chief Architect</p>
                </div>
              </div>

              <div className="space-y-6 pt-4">
                <div>
                  <p className="text-[10px] font-black uppercase tracking-widest text-slate-300 mb-1">Email Address</p>
                  <p className="font-bold text-sm text-primary">alex.wright@architect.studio</p>
                </div>
                <div>
                  <p className="text-[10px] font-black uppercase tracking-widest text-slate-300 mb-1">Phone Number</p>
                  <p className="font-bold text-sm text-primary">+1 (555) 892-0441</p>
                </div>
                <div>
                  <p className="text-[10px] font-black uppercase tracking-widest text-slate-300 mb-1">Office Location</p>
                  <p className="font-bold text-sm text-primary">San Francisco, CA</p>
                </div>
              </div>
            </div>
          </section>

          <section className="bg-slate-50/50 p-8 rounded-3xl border border-slate-100">
            <h3 className="font-headline text-lg font-bold text-primary mb-8">Security & Settings</h3>
            <div className="space-y-3">
              {[
                { icon: Lock, label: 'Update Password' },
                { icon: Bell, label: 'Notification Rules' },
                { icon: CreditCard, label: 'Payment Methods' },
              ].map((item) => (
                <button 
                  key={item.label}
                  className="w-full flex items-center justify-between p-4 rounded-2xl hover:bg-white hover:shadow-sm transition-all group"
                >
                  <div className="flex items-center gap-4">
                    <item.icon className="w-5 h-5 text-primary" />
                    <span className="text-sm font-bold text-slate-600 group-hover:text-primary">{item.label}</span>
                  </div>
                  <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-primary" />
                </button>
              ))}
              <div className="w-full flex items-center justify-between p-4 rounded-2xl bg-white/50">
                <div className="flex items-center gap-4">
                  <ShieldCheck className="w-5 h-5 text-primary" />
                  <span className="text-sm font-bold text-slate-600">2FA Authorization</span>
                </div>
                <span className="px-2 py-1 bg-emerald-50 text-emerald-600 text-[8px] font-black rounded uppercase tracking-widest">Active</span>
              </div>
            </div>
          </section>
        </div>

        {/* Right Column */}
        <div className="col-span-12 lg:col-span-8 space-y-12">
          {/* Recently Viewed */}
          <section>
            <div className="flex justify-between items-end mb-8">
              <div>
                <h3 className="font-headline text-2xl font-black text-primary">Recently Viewed</h3>
                <p className="text-sm font-bold text-slate-400">Your latest curated property explorations</p>
              </div>
              <button className="text-xs font-black text-primary uppercase tracking-widest border-b-2 border-primary pb-1">View All History</button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="group bg-white rounded-3xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-500 border border-slate-50">
                <div className="h-56 relative overflow-hidden">
                  <img 
                    alt="Modernist Villa" 
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" 
                    src="/static/house.jpg" 
                  />
                  <button className="absolute top-6 right-6 p-3 bg-white/90 backdrop-blur rounded-full text-secondary shadow-lg">
                    <Heart className="w-5 h-5 fill-secondary" />
                  </button>
                </div>
                <div className="p-8">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-headline font-black text-xl text-primary">The Glass Pavilion</h4>
                    <p className="font-headline font-black text-primary">$4,250,000</p>
                  </div>
                  <p className="text-sm font-bold text-slate-400 mb-6">Atherton, California</p>
                  <div className="flex gap-6 text-[10px] font-black text-slate-300 uppercase tracking-widest border-t border-slate-50 pt-6">
                    <span className="flex items-center gap-2"><Square className="w-4 h-4" /> 5,200 sqft</span>
                    <span className="flex items-center gap-2"><Bed className="w-4 h-4" /> 4 Beds</span>
                    <span className="flex items-center gap-2"><Bath className="w-4 h-4" /> 5 Baths</span>
                  </div>
                </div>
              </div>

              <div className="group bg-white rounded-3xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-500 border border-slate-50">
                <div className="h-56 relative overflow-hidden">
                  <img 
                    alt="Penthouse View" 
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" 
                    src="/static/house.jpg" 
                  />
                  <button className="absolute top-6 right-6 p-3 bg-white/20 backdrop-blur rounded-full text-white">
                    <Heart className="w-5 h-5" />
                  </button>
                </div>
                <div className="p-8">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-headline font-black text-xl text-primary">Summit Heights Unit 42</h4>
                    <p className="font-headline font-black text-primary">$2,890,000</p>
                  </div>
                  <p className="text-sm font-bold text-slate-400 mb-6">San Francisco, CA</p>
                  <div className="flex gap-6 text-[10px] font-black text-slate-300 uppercase tracking-widest border-t border-slate-50 pt-6">
                    <span className="flex items-center gap-2"><Square className="w-4 h-4" /> 3,100 sqft</span>
                    <span className="flex items-center gap-2"><Bed className="w-4 h-4" /> 2 Beds</span>
                    <span className="flex items-center gap-2"><Bath className="w-4 h-4" /> 3 Baths</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Prediction History */}
          <section>
            <h3 className="font-headline text-2xl font-black text-primary mb-8">Prediction History</h3>
            <div className="space-y-4">
              {[
                { address: '124 Oakwood Drive, Palo Alto', date: 'Oct 14, 2024', accuracy: '98.4%', price: '$3.2M', predicted: '$3.54M' },
                { address: 'The Marquee Loft, SoMa District', date: 'Sep 28, 2024', accuracy: '96.2%', price: '$1.8M', predicted: '$1.92M' },
              ].map((item, idx) => (
                <div key={idx} className="bg-slate-50/50 rounded-3xl border-l-4 border-secondary p-8 flex flex-col md:flex-row md:items-center justify-between gap-8 hover:bg-white hover:shadow-sm transition-all border border-slate-100">
                  <div className="flex items-center gap-6">
                    <div className="w-16 h-16 rounded-2xl bg-white flex items-center justify-center text-secondary shadow-sm">
                      <TrendingUp className="w-8 h-8" />
                    </div>
                    <div>
                      <h4 className="font-bold text-primary">{item.address}</h4>
                      <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mt-1">Analyzed {item.date} • Accuracy: {item.accuracy}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-12">
                    <div className="text-right">
                      <p className="text-[10px] font-black uppercase tracking-widest text-slate-300 mb-1">Listing Price</p>
                      <p className="font-headline font-black text-xl text-primary">{item.price}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] font-black uppercase tracking-widest text-secondary mb-1">Predicted (2025)</p>
                      <p className="font-headline font-black text-xl text-secondary">{item.predicted}</p>
                    </div>
                    <button className="p-3 hover:bg-white rounded-full transition-colors text-slate-300 hover:text-primary">
                      <ChevronRight className="w-6 h-6" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Saved Searches & Support */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
            <section>
              <h3 className="font-headline text-lg font-bold text-primary mb-6">Saved Searches</h3>
              <div className="flex flex-wrap gap-3">
                <div className="px-5 py-2.5 bg-primary text-white rounded-full text-[10px] font-black uppercase tracking-widest flex items-center gap-3 shadow-lg shadow-primary/20">
                  Modern San Francisco <X className="w-3 h-3 cursor-pointer" />
                </div>
                <div className="px-5 py-2.5 bg-slate-50 text-slate-500 rounded-full text-[10px] font-black uppercase tracking-widest flex items-center gap-3 border border-slate-100 hover:bg-primary hover:text-white transition-all cursor-pointer group">
                  Palo Alto 4+ Bed <Bell className="w-3 h-3 group-hover:text-white" />
                </div>
                <div className="px-5 py-2.5 bg-slate-50 text-slate-500 rounded-full text-[10px] font-black uppercase tracking-widest flex items-center gap-3 border border-slate-100 hover:bg-primary hover:text-white transition-all cursor-pointer group">
                  Waterfront 3M+ <Bell className="w-3 h-3 group-hover:text-white" />
                </div>
              </div>
            </section>

            <section className="bg-primary text-white p-8 rounded-3xl flex flex-col justify-center relative overflow-hidden">
              <div className="relative z-10">
                <h3 className="font-headline text-lg font-bold mb-2">Architectural Support</h3>
                <p className="text-xs text-white/60 mb-6 leading-relaxed">Need help navigating deep market insights or custom API access?</p>
                <div className="flex gap-6">
                  <button className="text-[10px] font-black uppercase tracking-widest border-b border-white/20 pb-1 hover:border-white transition-all">Knowledge Base</button>
                  <button className="text-[10px] font-black uppercase tracking-widest border-b border-white/20 pb-1 hover:border-white transition-all">Contact Specialist</button>
                </div>
              </div>
              <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2"></div>
            </section>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
