import React from 'react';
import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="bg-slate-50 border-t border-slate-200 mt-24">
      <div className="max-w-7xl mx-auto px-8 py-16 grid grid-cols-1 md:grid-cols-4 gap-12">
        <div className="space-y-6">
          <div className="text-xl font-black text-primary uppercase font-headline">PredictMyHome</div>
          <p className="text-sm text-slate-600 leading-relaxed">
            The intersection of architectural excellence and quantitative investment analysis. Redefining the high-end property market.
          </p>
        </div>
        
        <div>
          <h5 className="text-xs uppercase tracking-widest font-bold text-primary mb-6">Platform</h5>
          <ul className="space-y-4">
            <li><Link to="/search" className="text-slate-600 hover:text-primary text-sm transition-all">Market Reports</Link></li>
            <li><Link to="/predict" className="text-slate-600 hover:text-primary text-sm transition-all">Price Prediction Tool</Link></li>
            <li><Link to="/services" className="text-slate-600 hover:text-primary text-sm transition-all">Architectural Directory</Link></li>
          </ul>
        </div>

        <div>
          <h5 className="text-xs uppercase tracking-widest font-bold text-primary mb-6">Company</h5>
          <ul className="space-y-4">
            <li><Link to="/about" className="text-slate-600 hover:text-primary text-sm transition-all">About Us</Link></li>
            <li><Link to="/contact" className="text-slate-600 hover:text-primary text-sm transition-all">Contact Us</Link></li>
            <li><Link to="#" className="text-slate-600 hover:text-primary text-sm transition-all">Privacy Policy</Link></li>
          </ul>
        </div>

        <div>
          <h5 className="text-xs uppercase tracking-widest font-bold text-primary mb-6">Newsletter</h5>
          <p className="text-sm text-slate-600 mb-4">Market intelligence delivered weekly.</p>
          <div className="flex rounded-lg overflow-hidden border border-slate-200 bg-white">
            <input className="flex-1 px-4 py-2 text-xs focus:ring-0 border-none bg-transparent" placeholder="Email Address" type="email" />
            <button className="bg-primary text-white px-4 text-xs font-bold hover:opacity-90">Join</button>
          </div>
        </div>
      </div>
      
      <div className="max-w-7xl mx-auto px-8 py-8 border-t border-slate-200 flex flex-col md:flex-row justify-between items-center gap-4">
        <div>
          <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">© 2026 PredictMyHome. Meticulously engineered for precision.</p>
          <p className="text-[10px] uppercase tracking-widest text-slate-400 font-semibold mt-1">This website is for educational purposes only.</p>
        </div>
        <div className="flex space-x-8">
          <Link to="#" className="text-[10px] uppercase font-bold text-slate-500 hover:text-primary">Terms of Service</Link>
          <Link to="#" className="text-[10px] uppercase font-bold text-slate-500 hover:text-primary">Cookie Settings</Link>
        </div>
      </div>
    </footer>
  );
}
