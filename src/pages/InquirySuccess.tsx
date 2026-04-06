import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { 
  CheckCircle2, 
  LayoutDashboard, 
  Search, 
  BarChart3, 
  HelpCircle, 
  ArrowRight,
  User,
  Heart,
  Bell,
  Settings,
  LogOut
} from 'lucide-react';

export default function InquirySuccess() {
  const navigate = useNavigate();

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex min-h-screen bg-white"
    >
      {/* Sidebar */}
      <aside className="hidden lg:flex flex-col w-72 border-r border-slate-100 bg-slate-50/50 p-8">
        <div className="flex items-center gap-4 mb-12">
          <div className="w-12 h-12 rounded-full bg-primary-container flex items-center justify-center text-primary font-bold text-lg">
            AA
          </div>
          <div>
            <h3 className="font-headline font-bold text-primary text-sm leading-tight">Azure Account</h3>
            <p className="text-[10px] text-slate-400 uppercase tracking-widest mt-1">Premium Architect</p>
          </div>
        </div>

        <nav className="flex-1 space-y-2">
          {[
            { icon: LayoutDashboard, label: 'Dashboard', active: false },
            { icon: Heart, label: 'Saved Homes', active: false },
            { icon: Bell, label: 'Price Alerts', active: false },
            { icon: Settings, label: 'Account Settings', active: false },
            { icon: HelpCircle, label: 'Support', active: false },
          ].map((item) => (
            <button 
              key={item.label}
              className={`flex items-center gap-4 w-full px-4 py-3 rounded-xl text-sm font-bold transition-all ${
                item.active ? 'bg-white text-primary shadow-sm' : 'text-slate-400 hover:text-primary hover:bg-white/50'
              }`}
            >
              <item.icon className="w-5 h-5" />
              {item.label}
            </button>
          ))}
        </nav>

        <div className="mt-auto pt-8 border-t border-slate-100 space-y-6">
          <button className="w-full bg-primary text-white py-4 rounded-xl text-xs font-black uppercase tracking-widest hover:opacity-90 transition-all shadow-xl shadow-primary/20">
            Upgrade to Pro
          </button>
          <button className="flex items-center gap-4 px-4 text-sm font-bold text-red-500 hover:text-red-600 transition-colors">
            <LogOut className="w-5 h-5" />
            Logout
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8 md:p-16 lg:p-24 overflow-y-auto">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-16 items-start mb-24">
            <div className="lg:col-span-7">
              <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.2 }}
              >
                <span className="inline-block px-4 py-1.5 bg-emerald-50 text-emerald-600 text-[10px] font-black rounded-full tracking-widest uppercase mb-8">
                  Verified Submission
                </span>
                <h1 className="font-headline text-6xl md:text-7xl font-extrabold text-primary tracking-tighter mb-8 leading-[0.9]">
                  Inquiry Received
                </h1>
                <p className="text-slate-500 text-lg max-w-lg mb-12 leading-relaxed font-light">
                  Thank you for your interest. We've registered your request with our architectural consultants and price prediction engine.
                </p>
                <div className="flex flex-wrap gap-4">
                  <button 
                    onClick={() => navigate('/dashboard')}
                    className="bg-primary text-white px-10 py-5 rounded-2xl font-bold text-xs uppercase tracking-widest shadow-2xl shadow-primary/20 hover:translate-y-[-2px] active:scale-95 transition-all"
                  >
                    Return to Dashboard
                  </button>
                  <button 
                    onClick={() => navigate('/search')}
                    className="bg-white text-primary px-10 py-5 rounded-2xl font-bold text-xs uppercase tracking-widest border border-slate-100 hover:bg-slate-50 transition-all"
                  >
                    View Other Properties
                  </button>
                </div>
              </motion.div>
            </div>

            <div className="lg:col-span-5">
              <motion.div 
                initial={{ scale: 0.95, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="bg-white rounded-3xl p-10 shadow-[0_32px_64px_-12px_rgba(0,32,69,0.08)] border border-slate-50 relative"
              >
                <div className="flex justify-between items-start mb-12">
                  <div>
                    <p className="text-[10px] font-black text-slate-300 uppercase tracking-widest mb-2">Inquiry ID</p>
                    <p className="font-headline text-3xl font-black text-primary">#PH-2941-AZ</p>
                  </div>
                  <div className="h-14 w-14 rounded-full bg-emerald-50 flex items-center justify-center text-emerald-500">
                    <CheckCircle2 className="w-8 h-8" />
                  </div>
                </div>

                <div className="space-y-8">
                  <div className="flex gap-6">
                    <div className="w-1 bg-emerald-500 rounded-full"></div>
                    <div>
                      <h4 className="font-headline font-bold text-primary text-sm">Expert Review</h4>
                      <p className="text-xs text-slate-400 mt-2 leading-relaxed">An architectural specialist will contact you within 24 hours.</p>
                    </div>
                  </div>
                  <div className="flex gap-6">
                    <div className="w-1 bg-slate-100 rounded-full"></div>
                    <div>
                      <h4 className="font-headline font-bold text-slate-400 text-sm">Data Analysis</h4>
                      <p className="text-xs text-slate-400 mt-2 leading-relaxed">Generating custom market valuation report for your inquiry.</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="md:col-span-2 bg-slate-50/50 p-10 rounded-3xl border-l-4 border-emerald-500 flex flex-col justify-between min-h-[240px]">
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <BarChart3 className="text-emerald-500 w-6 h-6" />
                  <h3 className="font-headline font-bold text-primary text-lg">Predictive Advantage</h3>
                </div>
                <p className="text-sm text-slate-500 leading-relaxed max-w-xl">
                  While you wait, our AI is processing 42 market variables to ensure you receive the most accurate architectural appraisal.
                </p>
              </div>
              <div className="flex items-center gap-6 mt-8">
                <div className="flex -space-x-3">
                  {[1, 2, 3].map((i) => (
                    <img 
                      key={i}
                      className="w-10 h-10 rounded-full border-2 border-white object-cover" 
                      src="/static/house.jpg"
                      alt="Agent"
                    />
                  ))}
                  <div className="w-10 h-10 rounded-full bg-primary-container flex items-center justify-center text-[10px] font-bold text-primary border-2 border-white">
                    +12
                  </div>
                </div>
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Agents Online Now</p>
              </div>
            </div>

            <div className="bg-white p-10 rounded-3xl border border-slate-100 flex flex-col items-center text-center shadow-sm">
              <div className="h-16 w-16 bg-slate-50 rounded-full flex items-center justify-center mb-8">
                <HelpCircle className="text-primary w-8 h-8" />
              </div>
              <h3 className="font-headline font-bold text-primary mb-3">Need Immediate Help?</h3>
              <p className="text-xs text-slate-400 mb-8 leading-relaxed">Our support team is available 24/7 for premium members.</p>
              <button className="text-primary font-black text-[10px] uppercase tracking-widest hover:text-emerald-500 transition-all inline-flex items-center gap-3 group">
                Go to Support 
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>

          <footer className="mt-32 pt-12 border-t border-slate-100 flex flex-col md:flex-row justify-between items-center gap-8 opacity-40">
            <div className="flex items-center gap-6">
              <span className="font-headline font-black text-xl tracking-tighter text-primary">PredictMyHome</span>
              <div className="h-4 w-[1px] bg-slate-300"></div>
              <p className="text-[10px] uppercase font-black tracking-widest">© 2026 Azure Estate Group</p>
            </div>
            <div className="flex gap-10">
              <button className="text-[10px] uppercase font-black tracking-widest hover:text-primary transition-colors">Privacy Protocol</button>
              <button className="text-[10px] uppercase font-black tracking-widest hover:text-primary transition-colors">Architectural Terms</button>
            </div>
          </footer>
        </div>
      </main>
    </motion.div>
  );
}
