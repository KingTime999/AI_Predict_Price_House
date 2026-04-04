import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, Chrome, X } from 'lucide-react';

export default function Login() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    // Simulate login
    setTimeout(() => {
      setLoading(false);
      navigate('/dashboard');
    }, 1000);
  };

  return (
    <div className="min-h-screen flex items-stretch">
      <section className="hidden lg:flex lg:w-3/5 architectural-gradient relative p-16 flex-col justify-between">
        <div>
          <span className="text-white font-headline text-2xl font-black tracking-tighter">PredictMyHome</span>
        </div>
        <div className="max-w-xl">
          <h1 className="font-headline text-5xl font-extrabold text-white tracking-tighter leading-tight mb-6">
            Engineering the future of <span className="text-secondary-container">real estate valuation.</span>
          </h1>
          <p className="text-white/80 text-lg leading-relaxed">
            Access high-fidelity market data and precision AI predictions for your next architectural investment.
          </p>
        </div>
        <div className="flex gap-4">
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl border border-white/10 flex-1">
            <p className="text-white/60 text-[10px] uppercase tracking-widest font-bold mb-1">Market Accuracy</p>
            <p className="text-white font-headline text-xl font-bold">99.4%</p>
          </div>
          <div className="bg-white/10 backdrop-blur-md p-4 rounded-xl border border-white/10 flex-1">
            <p className="text-white/60 text-[10px] uppercase tracking-widest font-bold mb-1">Active Insights</p>
            <p className="text-white font-headline text-xl font-bold">2.4M+</p>
          </div>
        </div>
      </section>

      <section className="w-full lg:w-2/5 bg-white flex items-center justify-center p-8 relative">
        <Link to="/" className="absolute top-8 right-8 text-slate-400 hover:text-primary transition-colors">
          <X className="w-6 h-6" />
        </Link>
        <div className="w-full max-w-sm">
          <header className="mb-10">
            <h2 className="font-headline text-3xl font-extrabold text-primary tracking-tight mb-2">Welcome Back</h2>
            <p className="text-slate-500 text-sm">Please enter your credentials to access the platform.</p>
          </header>
          <form className="space-y-6" onSubmit={handleSubmit}>
            <div className="space-y-1.5">
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider">Email Address</label>
              <input 
                className="w-full bg-slate-50 border-none rounded-lg px-4 py-3.5 text-sm focus:ring-2 focus:ring-primary/20 transition-all" 
                placeholder="architect@firm.com" 
                type="email" 
                required
              />
            </div>
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider">Password</label>
                <a className="text-xs font-semibold text-secondary hover:underline" href="#">Forgot?</a>
              </div>
              <input 
                className="w-full bg-slate-50 border-none rounded-lg px-4 py-3.5 text-sm focus:ring-2 focus:ring-primary/20 transition-all" 
                placeholder="••••••••" 
                type="password" 
                required
              />
            </div>
            <button 
              className="w-full bg-primary text-white font-headline font-bold py-4 rounded-xl shadow-lg hover:opacity-90 transition-all disabled:opacity-50" 
              type="submit"
              disabled={loading}
            >
              {loading ? 'Authenticating...' : 'Login to Dashboard'}
            </button>
            <div className="relative flex items-center py-4">
              <div className="flex-grow border-t border-slate-100"></div>
              <span className="flex-shrink mx-4 text-xs font-bold text-slate-400 uppercase tracking-widest">or</span>
              <div className="flex-grow border-t border-slate-100"></div>
            </div>
            <button className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors text-sm font-semibold text-primary" type="button">
              <Chrome className="w-5 h-5" />
              Google
            </button>
          </form>
          <footer className="mt-12 text-center">
            <p className="text-sm text-slate-500">
              New to the platform? <Link to="/register" className="text-primary font-bold hover:underline ml-1">Register Account</Link>
            </p>
          </footer>
        </div>
      </section>
    </div>
  );
}
