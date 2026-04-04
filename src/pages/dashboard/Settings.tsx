import React from 'react';
import { motion } from 'motion/react';
import { 
  Lock, 
  CreditCard, 
  ShieldCheck, 
  ChevronRight,
  Laptop,
  Smartphone,
  CheckCircle2,
  Bell,
  TrendingDown,
  FileText,
  Activity,
  Trash2
} from 'lucide-react';

export default function Settings() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-12"
    >
      <header>
        <p className="font-bold text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-2">Architectural Portal</p>
        <h2 className="font-headline text-4xl font-black tracking-tight text-primary">Account Settings</h2>
        <p className="text-slate-400 font-bold text-sm mt-2">Manage your architectural data preferences and security protocols.</p>
      </header>

      <div className="grid grid-cols-12 gap-10">
        {/* Left Column */}
        <div className="col-span-12 lg:col-span-7 space-y-10">
          {/* Change Password */}
          <section className="bg-white p-10 rounded-3xl shadow-[0_20px_50px_rgba(0,0,0,0.02)] border border-slate-50 border-l-4 border-primary">
            <div className="flex items-center gap-4 mb-10">
              <div className="w-12 h-12 rounded-2xl bg-slate-50 flex items-center justify-center text-primary">
                <Lock className="w-6 h-6" />
              </div>
              <h3 className="font-headline text-xl font-bold text-primary">Change Password</h3>
            </div>
            
            <form className="space-y-8">
              <div className="space-y-2">
                <label className="text-[10px] font-black uppercase tracking-widest text-slate-300 ml-1">Current Password</label>
                <input 
                  className="w-full bg-slate-50 border-none rounded-2xl px-6 py-4 focus:ring-2 focus:ring-primary/5 transition-all text-sm font-bold" 
                  placeholder="••••••••" 
                  type="password" 
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-slate-300 ml-1">New Password</label>
                  <input 
                    className="w-full bg-slate-50 border-none rounded-2xl px-6 py-4 focus:ring-2 focus:ring-primary/5 transition-all text-sm font-bold" 
                    placeholder="••••••••" 
                    type="password" 
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-slate-300 ml-1">Confirm New Password</label>
                  <input 
                    className="w-full bg-slate-50 border-none rounded-2xl px-6 py-4 focus:ring-2 focus:ring-primary/5 transition-all text-sm font-bold" 
                    placeholder="••••••••" 
                    type="password" 
                  />
                </div>
              </div>
              <div className="pt-4">
                <button className="bg-primary text-white px-10 py-5 rounded-2xl font-bold text-xs uppercase tracking-widest transition-all hover:shadow-2xl hover:shadow-primary/20 active:scale-95">
                  Update Credentials
                </button>
              </div>
            </form>
          </section>

          {/* Premium Subscription */}
          <section className="bg-white p-10 rounded-3xl shadow-[0_20px_50px_rgba(0,0,0,0.02)] border border-slate-50 border-l-4 border-secondary">
            <div className="flex items-center justify-between mb-10">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-2xl bg-emerald-50 flex items-center justify-center text-secondary">
                  <CreditCard className="w-6 h-6" />
                </div>
                <h3 className="font-headline text-xl font-bold text-primary">Premium Subscription</h3>
              </div>
              <span className="bg-emerald-50 text-secondary text-[10px] font-black px-4 py-1.5 rounded-full uppercase tracking-widest">Pro Plan</span>
            </div>

            <div className="bg-slate-50/50 rounded-3xl p-8 mb-8 border border-slate-100">
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-8 bg-primary rounded-lg flex items-center justify-center">
                    <div className="w-8 h-1 bg-white/20 rounded-full"></div>
                  </div>
                  <div>
                    <p className="font-black text-primary">•••• 4242</p>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Exp: 12/26</p>
                  </div>
                </div>
              </div>
              <div className="flex gap-6">
                <button className="text-[10px] font-black text-primary uppercase tracking-widest hover:underline">Edit Method</button>
                <button className="text-[10px] font-black text-red-500 uppercase tracking-widest hover:underline">Remove</button>
              </div>
            </div>

            <div className="flex items-center justify-between pt-8 border-t border-slate-100">
              <div>
                <p className="text-[10px] font-black uppercase tracking-widest text-slate-300 mb-1">Next billing cycle</p>
                <p className="text-sm font-black text-primary">Oct 12, 2024 • $29.00/mo</p>
              </div>
              <button className="text-[10px] font-black text-secondary uppercase tracking-widest hover:underline">Manage Plan</button>
            </div>
          </section>
        </div>

        {/* Right Column */}
        <div className="col-span-12 lg:col-span-5 space-y-10">
          {/* Security Protocols */}
          <section className="bg-slate-50/50 p-10 rounded-3xl border border-slate-100">
            <div className="flex items-center gap-4 mb-10">
              <div className="w-12 h-12 rounded-2xl bg-white flex items-center justify-center text-primary shadow-sm">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <h3 className="font-headline text-xl font-bold text-primary">Security Protocols</h3>
            </div>

            <div className="bg-white p-6 rounded-2xl mb-10 flex items-center justify-between border border-slate-50 shadow-sm">
              <div>
                <p className="font-black text-sm text-primary">Two-Factor Authentication</p>
                <p className="text-xs font-bold text-slate-400">Recommended for high-value accounts</p>
              </div>
              <button className="w-14 h-7 rounded-full bg-secondary relative flex items-center px-1 transition-all">
                <div className="h-5 w-5 bg-white rounded-full ml-auto shadow-sm"></div>
              </button>
            </div>

            <div>
              <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-300 mb-6 ml-1">Active Sessions</h4>
              <div className="space-y-6">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-white flex items-center justify-center text-slate-400 shadow-sm">
                    <Laptop className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-sm font-bold text-primary">MacBook Pro 16" • London, UK</p>
                    <p className="text-[10px] font-black text-secondary uppercase tracking-widest">Active now</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-white flex items-center justify-center text-slate-400 shadow-sm">
                    <Smartphone className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-sm font-bold text-primary">iPhone 15 Pro • Paris, FR</p>
                    <p className="text-[10px] font-black text-slate-300 uppercase tracking-widest">2 hours ago</p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Communication Pulse */}
          <section className="bg-white p-10 rounded-3xl shadow-[0_20px_50px_rgba(0,0,0,0.02)] border border-slate-50 border-t-4 border-primary">
            <div className="flex items-center gap-4 mb-10">
              <div className="w-12 h-12 rounded-2xl bg-slate-50 flex items-center justify-center text-primary">
                <Bell className="w-6 h-6" />
              </div>
              <h3 className="font-headline text-xl font-bold text-primary">Communication Pulse</h3>
            </div>

            <div className="space-y-8">
              {[
                { label: 'New Matches', desc: 'Instant property alerts', active: true },
                { label: 'Price Drops', desc: 'Value decrease alerts', active: true },
                { label: 'Market Reports', desc: 'Weekly analytical deep-dives', active: false },
                { label: 'Account Activity', desc: 'Logins and security logs', active: true },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-black text-primary">{item.label}</p>
                    <p className="text-xs font-bold text-slate-400">{item.desc}</p>
                  </div>
                  <button className={`w-12 h-6 rounded-full relative flex items-center px-1 transition-all ${item.active ? 'bg-secondary' : 'bg-slate-100'}`}>
                    <div className={`h-4 w-4 bg-white rounded-full shadow-sm transition-all ${item.active ? 'ml-auto' : 'ml-0'}`}></div>
                  </button>
                </div>
              ))}
            </div>
          </section>

          <footer className="pt-10 flex justify-center">
            <button className="text-[10px] font-black text-red-500 uppercase tracking-widest hover:underline flex items-center gap-2">
              <Trash2 className="w-4 h-4" />
              Deactivate Account
            </button>
          </footer>
        </div>
      </div>
    </motion.div>
  );
}
