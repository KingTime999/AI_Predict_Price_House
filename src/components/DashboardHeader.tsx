import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Bell, User, Search } from 'lucide-react';

export default function DashboardHeader() {
  const location = useLocation();

  return (
    <header className="w-full top-0 sticky z-40 bg-white/80 backdrop-blur-md border-b border-slate-100">
      <div className="max-w-7xl mx-auto px-8 h-20 flex items-center justify-between">
        <Link to="/" className="text-xl font-headline font-black tracking-tighter text-primary uppercase">
          PredictMyHome
        </Link>
        
        <div className="flex items-center gap-10">
          <nav className="hidden md:flex items-center gap-8 font-headline tracking-tight text-sm font-bold">
            <Link to="/search" className="text-slate-500 hover:text-primary transition-colors">Market Insights</Link>
            <Link to="/search" className="text-slate-500 hover:text-primary transition-colors">Properties</Link>
            <Link to="/predict" className="text-slate-500 hover:text-primary transition-colors">Predictions</Link>
            <Link to="/services" className="text-slate-500 hover:text-primary transition-colors">Tools</Link>
          </nav>

          <div className="flex items-center gap-6">
            <div className="relative hidden sm:block">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300" />
              <input 
                className="bg-slate-50 border border-slate-100 rounded-full px-10 py-2 text-xs w-64 focus:ring-2 focus:ring-primary/5 outline-none font-medium" 
                placeholder="Search property ID..." 
                type="text" 
              />
            </div>
            <button className="text-slate-400 hover:text-primary transition-colors relative">
              <Bell className="w-5 h-5" />
              <span className="absolute -top-1 -right-1 w-2 h-2 bg-secondary rounded-full border-2 border-white"></span>
            </button>
            <button className="text-slate-400 hover:text-primary transition-colors">
              <User className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
